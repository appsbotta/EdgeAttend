import socket
import threading
import struct
import time
import json
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import cv2


HOST                = "0.0.0.0"
PORT                = 9999
MJPEG_PORT          = 8080
SERVER_DISPLAY_NAME = "Host"
TILE_W, TILE_H      = 400, 300
MAX_COLS            = 3
GRID_FPS            = 8
GRID_JPEG_QUALITY   = 100

MOOD_COLORS = {
    "Attentive":     (50, 220, 130),
    "Non-Attentive": (0,  50,  220),
    "Loading...":    (180, 180, 180),
}

MSG_FRAME = b'F'
MSG_MOOD  = b'M'
MSG_GRID  = b'G'
MSG_ATTN  = b'A'


def _label_to_color(label: str) -> tuple:
    """
    Maps an attentiveness label string to its corresponding BGR color tuple.
    Falls back to a neutral grey if the label is not in the MOOD_COLORS table.

    Args:
        label: A string like "Attentive", "Non-Attentive", or "Loading...".

    Returns:
        A (B, G, R) tuple used for coloring UI elements in OpenCV.
    """
    return MOOD_COLORS.get(label, (180, 180, 180))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """
    Reads exactly n bytes from a socket, blocking until all bytes arrive.
    Handles partial reads by looping until the full requested size is collected.

    Args:
        conn: An active socket connection to read from.
        n: The exact number of bytes to read.

    Returns:
        A bytes object of length n.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_typed(conn: socket.socket):
    """
    Reads one complete typed message from the socket using a 5-byte header.
    The header encodes the message type (1 byte) and payload length (4 bytes, big-endian).

    Args:
        conn: An active socket connection to read from.

    Returns:
        A tuple of (message_type, payload) where message_type is a single byte
        and payload is the raw bytes of the message body (may be empty).
    """
    hdr    = _recv_exact(conn, 5)
    mtype  = hdr[0:1]
    length = struct.unpack(">I", hdr[1:])[0]
    data   = _recv_exact(conn, length) if length else b""
    return mtype, data


def send_typed(conn: socket.socket, mtype: bytes, data: bytes) -> None:
    """
    Sends a typed message over a socket with a 5-byte framing header.
    Prepends the message type byte and a 4-byte big-endian length before the payload.

    Args:
        conn: An active socket connection to write to.
        mtype: A single byte identifying the message type (e.g. MSG_FRAME, MSG_GRID).
        data: The raw bytes payload to send.

    Returns:
        None
    """
    conn.sendall(mtype + struct.pack(">I", len(data)) + data)


clients      : dict = {}
clients_lock = threading.Lock()

_srv      = {
    "frame": None,
    "label": "Host",
    "color": (180, 180, 180),
    "score": 0.0,
}
_srv_lock = threading.Lock()

_grid_jpeg_annotated      : bytes = b""
_grid_jpeg_annotated_lock = threading.Lock()

_grid_jpeg_clean      : bytes = b""
_grid_jpeg_clean_lock = threading.Lock()


class ClientHandler(threading.Thread):
    """
    Spawns one background thread per connected client.
    Listens for incoming frame (F), attentiveness (A), and control (M) messages
    and updates the shared clients dict accordingly.
    """

    def __init__(self, conn: socket.socket, addr, cid: str):
        """
        Initialises the handler with connection details and a unique client ID.
        Sets up a per-connection lock to protect concurrent socket writes.

        Args:
            conn: The accepted socket connection for this client.
            addr: The (host, port) address tuple of the remote client.
            cid: A short string identifier assigned to this client (e.g. "C01").

        Returns:
            None
        """
        super().__init__(daemon=True, name=f"H-{cid}")
        self.conn      = conn
        self.addr      = addr
        self.cid       = cid
        self.conn_lock = threading.Lock()

    def run(self):
        """
        Main loop for the client thread - registers the client, sends an ID handshake,
        then continuously reads and dispatches incoming typed messages until disconnect.
        Handles NAME updates, attentiveness labels, and raw video frames.

        Args:
            None (uses instance attributes set in __init__)

        Returns:
            None
        """
        cid = self.cid
        print(f"[+] {cid} connected from {self.addr}")

        with clients_lock:
            clients[cid] = {
                "frame":     None,
                "label":     "Loading...",
                "color":     _label_to_color("Loading..."),
                "score":     0.0,
                "name":      cid,
                "addr":      f"{self.addr[0]}:{self.addr[1]}",
                "last_seen": time.monotonic(),
                "conn":      self.conn,
                "lock":      self.conn_lock,
            }

        try:
            with self.conn_lock:
                send_typed(self.conn, MSG_MOOD, f"ID:{cid}".encode())
        except Exception as e:
            print(f"[-] Handshake fail {cid}: {e}")
            self._cleanup()
            return

        try:
            while True:
                mtype, payload = recv_typed(self.conn)

                if mtype == MSG_MOOD and payload.startswith(b"NAME:"):
                    name = payload[5:].decode(errors="replace").strip()
                    if name:
                        with clients_lock:
                            if cid in clients:
                                clients[cid]["name"] = name[:40]
                    continue

                if mtype == MSG_ATTN and payload:
                    msg = payload.decode(errors="replace").strip()
                    if ":" in msg:
                        lbl, _, sco_str = msg.partition(":")
                        lbl = lbl.strip()
                        try:
                            sco = float(sco_str.strip())
                        except ValueError:
                            sco = 0.0
                        with clients_lock:
                            if cid in clients:
                                clients[cid].update(label=lbl, color=_label_to_color(lbl), score=sco,)
                    continue

                if mtype == MSG_FRAME and payload:
                    frame = cv2.imdecode(
                        np.frombuffer(payload, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        with clients_lock:
                            if cid in clients:
                                clients[cid]["frame"]     = frame
                                clients[cid]["last_seen"] = time.monotonic()
                    continue

        except (ConnectionError, BrokenPipeError, OSError) as e:
            print(f"[-] {cid} disconnected: {e}")
        except Exception as e:
            print(f"[-] {cid} error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """
        Closes the socket and removes this client from the global clients dict.
        Safe to call multiple times - errors on close are silently ignored.
        """
        try:
            self.conn.close()
        except Exception:
            pass
        with clients_lock:
            clients.pop(self.cid, None)
        print(f"[-] {self.cid} removed.")


def _tile_annotated(frame, cid, name, label, color, score) -> np.ndarray:
    """
    Builds a single video tile with full mood annotation for the server-side display.
    Draws a name/label bar at the bottom and a coloured border reflecting attentiveness.

    Args:
        frame: A BGR numpy array from the client's camera, or None if not yet received.
        cid: The client ID string; an empty string signals this is the host tile.
        name: The display name shown in the bottom bar.
        label: The current attentiveness label (e.g. "Attentive").
        color: A (B, G, R) tuple used for the border and label text.
        score: A float confidence score from the attentiveness model (unused visually here).

    Returns:
        A (TILE_H x TILE_W x 3) uint8 numpy array ready to drop into the grid.
    """
    tile = (cv2.resize(frame, (TILE_W, TILE_H))
            if frame is not None
            else np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

    if frame is None:
        cv2.putText(tile, "Connecting...", (TILE_W // 4, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 2)

    cv2.rectangle(tile, (0, TILE_H - 52), (TILE_W, TILE_H), (18, 18, 18), -1)
    header = f"{name} (Host)" if not cid else name
    cv2.putText(tile, header,
                (8, TILE_H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(tile, label,
                (8, TILE_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), color, 3)
    return tile


def _tile_clean(frame, cid, name) -> np.ndarray:
    """
    Builds a plain video tile with only a subtle name bar and no mood information.
    This version is pushed to clients so they can overlay their own local badge.

    Args:
        frame: A BGR numpy array from the camera, or None if not yet available.
        cid: The client ID; empty string means this is the host tile.
        name: The display name shown in the thin bottom bar.

    Returns:
        A (TILE_H x TILE_W x 3) uint8 numpy array suitable for the clean grid.
    """
    tile = (cv2.resize(frame, (TILE_W, TILE_H))
            if frame is not None
            else np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

    if frame is None:
        cv2.putText(tile, "Connecting...", (TILE_W // 4, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 2)

    cv2.rectangle(tile, (0, TILE_H - 28), (TILE_W, TILE_H), (18, 18, 18), -1)
    bar_label = f"{name} (Host)" if not cid else name
    cv2.putText(tile, bar_label,
                (8, TILE_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 190, 190), 1)
    return tile


def _assemble_grid(tiles: list) -> np.ndarray:
    """
    Arranges a flat list of tiles into a rectangular grid image.
    Pads the list with blank black tiles if needed to fill the last row completely.

    Args:
        tiles: A list of (TILE_H x TILE_W x 3) numpy arrays, one per participant.

    Returns:
        A single numpy array with all tiles arranged in rows of up to MAX_COLS columns.
        Returns a placeholder "Waiting for clients..." image if the list is empty.
    """
    if not tiles:
        blank = np.zeros((TILE_H, TILE_W * 2, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for clients...",
                    (30, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return blank

    n    = len(tiles)
    cols = min(MAX_COLS, n)
    rows = (n + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

    return np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])


def _header_bar(width: int, text: str, height: int = 38) -> np.ndarray:
    """
    Creates a thin dark banner image with centred white text for display above a grid.
    Used to show participant count, current time, and other session metadata.

    Args:
        width: The pixel width of the bar, should match the grid width.
        text: The string to render inside the bar.
        height: The pixel height of the bar (default 38).

    Returns:
        A (height x width x 3) uint8 numpy array representing the header banner.
    """
    hdr = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(hdr, text,
                (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    return hdr


def build_annotated_grid() -> np.ndarray:
    """
    Assembles the full mood-annotated grid image shown on the server monitor window.
    Snapshots the current host frame and all client frames, builds annotated tiles,
    then stacks them with a header showing participant count and timestamp.

    Args:
        None (reads from global _srv and clients dicts under their locks)

    Returns:
        A numpy image array with a header bar on top and all annotated tiles below.
    """
    with _srv_lock:
        sf, sl, sc, ss = (_srv["frame"], _srv["label"],
                          _srv["color"], _srv["score"])
    with clients_lock:
        snap = {k: dict(v) for k, v in clients.items()}

    tiles = [_tile_annotated(sf, "", SERVER_DISPLAY_NAME, sl, sc, ss)]
    for cid, info in sorted(snap.items()):
        tiles.append(_tile_annotated(
            info["frame"], cid, info.get("name", cid),
            info["label"], info["color"], info["score"],
        ))

    grid = _assemble_grid(tiles)
    n_c  = len(snap)
    attentive_count = sum(
        1 for v in snap.values() if v["label"] == "Attentive"
    )
    hdr_text = (
        f"  Meeting Room - {n_c+1} participants"
        f"  |  {time.strftime('%H:%M:%S')}"
    )
    return np.vstack([_header_bar(grid.shape[1], hdr_text), grid])


def build_clean_grid() -> np.ndarray:
    """
    Assembles the clean (unannotated) grid image that gets pushed to all clients.
    Contains only name bars - no mood labels, scores, or coloured borders.

    Args:
        None (reads from global _srv and clients dicts under their locks)

    Returns:
        A numpy image array with a header bar and plain video tiles for every participant.
    """
    with _srv_lock:
        sf = _srv["frame"]
    with clients_lock:
        snap = {k: dict(v) for k, v in clients.items()}

    tiles = [_tile_clean(sf, "", SERVER_DISPLAY_NAME)]
    for cid, info in sorted(snap.items()):
        tiles.append(_tile_clean(info["frame"], cid, info.get("name", cid)))

    grid = _assemble_grid(tiles)
    n_c  = len(snap)
    hdr_text = (
        f"  Meeting Room - {n_c+1} participants"
        f"  |  {time.strftime('%H:%M:%S')}")
    return np.vstack([_header_bar(grid.shape[1], hdr_text, height=30), grid])


def _grid_encoder_loop():
    """
    Background loop that rebuilds both grid images at GRID_FPS and stores them as JPEG bytes.
    Runs the annotated grid for the server browser monitor and the clean grid for client push.
    """
    params   = [cv2.IMWRITE_JPEG_QUALITY, GRID_JPEG_QUALITY]
    interval = 1.0 / GRID_FPS

    while True:
        t = time.monotonic()
        ann = build_annotated_grid()
        ok, buf = cv2.imencode(".jpg", ann, params)
        if ok:
            with _grid_jpeg_annotated_lock:
                globals()["_grid_jpeg_annotated"] = buf.tobytes()

        cln = build_clean_grid()
        ok, buf = cv2.imencode(".jpg", cln, params)
        if ok:
            with _grid_jpeg_clean_lock:
                globals()["_grid_jpeg_clean"] = buf.tobytes()

        gap = interval - (time.monotonic() - t)
        if gap > 0:
            time.sleep(gap)


def _grid_push_loop():
    """
    Background loop that broadcasts the latest clean grid JPEG to every connected client.
    Runs at GRID_FPS and silently skips clients whose sockets have gone away.
    """
    interval = 1.0 / GRID_FPS
    while True:
        time.sleep(interval)

        with _grid_jpeg_clean_lock:
            jpeg = _grid_jpeg_clean
        if not jpeg:
            continue

        with clients_lock:
            targets = [(info["conn"], info["lock"])
                       for info in clients.values()]

        for conn, lock in targets:
            try:
                with lock:
                    send_typed(conn, MSG_GRID, jpeg)
            except OSError:
                pass


def server_cam_loop():
    """
    Continuously captures frames from the server's local webcam and stores them in _srv.
    If no camera is available, the host tile will remain a blank black square.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[~] No server webcam found - host tile will stay blank.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[OK] Server webcam opened.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with _srv_lock:
            _srv["frame"] = frame
        time.sleep(1.0 / 15.0)


_INDEX_HTML = """\
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Meeting Room - Attentiveness Monitor</title>
<style>
  body {{ margin:0; background:#111; display:flex; flex-direction:column;
          align-items:center; font-family:monospace; color:#eee; }}
  h2   {{ margin:12px 0 4px; }}
  img  {{ max-width:100%; border:2px solid #333; }}
  small{{ color:#666; margin-bottom:8px; }}
</style></head>
<body>
  <h2>Meeting Room - Attentiveness Monitor</h2>
  <img src="/stream">
  <small>server {ip}:{port} &nbsp;|&nbsp; <a href="/status" style="color:#888">JSON status</a></small>
</body></html>
"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler that serves the browser monitor over three routes:
    / returns the HTML page, /stream streams MJPEG frames, /status returns JSON client info.
    """

    def log_message(self, *a):
        """
        Suppresses the default per-request log lines to keep the terminal tidy.
        Overrides BaseHTTPRequestHandler.log_message with a no-op.

        Args:
            *a: Ignored arguments passed by the base class.

        Returns:
            None
        """
        pass

    def do_GET(self):
        """
        Routes incoming GET requests to the correct handler based on the request path.
        Unknown paths receive a 404 response.
        """
        if   self.path == "/":       self._index()
        elif self.path == "/stream": self._stream()
        elif self.path == "/status": self._status()
        else:
            self.send_error(404)

    def _index(self):
        """
        Serves the static HTML monitor page with the server IP and port injected.
        Responds with a 200 and the full HTML body as UTF-8 bytes.
        """
        ip, port = self.server.server_address
        body = _INDEX_HTML.format(ip=ip, port=port).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        """
        Streams the annotated grid as a continuous multipart/x-mixed-replace MJPEG feed.
        Keeps the connection open and pushes a new JPEG frame at each GRID_FPS interval.
        """
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _grid_jpeg_annotated_lock:
                    j = _grid_jpeg_annotated
                if j:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\nContent-Length: "
                        + str(len(j)).encode()
                        + b"\r\n\r\n" + j + b"\r\n")
                    self.wfile.flush()
                time.sleep(1.0 / GRID_FPS)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _status(self):
        """
        Returns a JSON snapshot of all currently connected clients and their attentiveness state.
        Includes name, label, score, and IP address for each client.
        """
        with clients_lock:
            data = {
                cid: {
                    "name":  v.get("name", cid),
                    "label": v["label"],
                    "score": round(v["score"], 3),
                    "addr":  v["addr"],
                }
                for cid, v in clients.items()
            }
        body = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_mjpeg_server():
    """
    Starts the MJPEG HTTP server on MJPEG_PORT in a background daemon thread.
    The server exposes the annotated grid stream and JSON status endpoint to browsers.
    """
    httpd = HTTPServer((HOST, MJPEG_PORT), _MJPEGHandler)
    threading.Thread(target=httpd.serve_forever,
                     daemon=True, name="MJPEG").start()
    ip = socket.gethostbyname(socket.gethostname())


def display_loop():
    """
    Opens a named OpenCV window and renders the annotated grid at GRID_FPS.
    Keeps running until the user presses Q, at which point it cleans up all windows.
    """
    win = "Meeting Room - Server Monitor  (Q to quit)"
    try:
        cv2.startWindowThread()
    except Exception:
        pass
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TILE_W * MAX_COLS, TILE_H * 2 + 50)

    print("[OK] Server display window open.")
    while True:
        grid = build_annotated_grid()
        cv2.imshow(win, grid)
        key = cv2.waitKey(max(1, int(1000 / GRID_FPS))) & 0xFF
        if key == ord("q"):
            print("[!] Q pressed - shutting down server.")
            break

    cv2.destroyAllWindows()


def accept_loop(srv: socket.socket):
    """
    Listens for incoming TCP connections and spawns a ClientHandler thread for each one.
    Assigns sequential IDs (C01, C02, ...) and enables TCP keep-alive on each socket.

    Args:
        srv: The bound and listening server socket to accept connections from.

    Returns:
        None - runs until the socket is closed or an OSError is raised.
    """
    n = 1
    while True:
        try:
            conn, addr = srv.accept()
        except OSError:
            break
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        cid = f"C{n:02d}"
        ClientHandler(conn, addr, cid).start()
        n += 1


def main():
    """
    Entry point - prompts for a host display name, binds the TCP socket,
    and starts all background threads (grid encoder, grid push, webcam, MJPEG server, accept loop).
    Blocks on the display window until Q is pressed or Ctrl+C is received, then shuts down cleanly.
    """
    global SERVER_DISPLAY_NAME

    try:
        host_name = input("Enter host display name [Host]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Aborted.")
        return
    if host_name:
        SERVER_DISPLAY_NAME = host_name[:40]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(16)

    ip = socket.gethostbyname(socket.gethostname())
    print(f"TCP server  →  {ip}:{PORT}")
    print(f"Clients connect to:  {ip}\n")

    for target, name in [
        (_grid_encoder_loop, "GridEncoder"),
        (_grid_push_loop,    "GridPush"),
        (server_cam_loop,    "ServerCam"),
    ]:
        threading.Thread(target=target, daemon=True, name=name).start()

    start_mjpeg_server()

    threading.Thread(target=accept_loop, args=(srv,), daemon=True, name="Accept").start()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),)

    print("Press Q in the display window or Ctrl+C to stop.\n")
    try:
        display_loop()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        srv.close()
        print("Server stopped.")


if __name__ == "__main__":
    main()