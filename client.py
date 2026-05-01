import socket
import threading
import struct
import time
import sys
import os
from collections import deque
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


PORT            = 9999
SEND_FPS        = 15
JPEG_QUALITY    = 100
CAM_INDEX       = 0
CAM_W, CAM_H    = 640, 480
MODEL_PATH_PTH  = "attentive_model_quantized.pth"
IMG_SIZE        = 160
INFER_EVERY_N   = 5
BATCH_SIZE      = 5

MOOD_COLORS = {
    "Attentive":      (50, 220, 130),
    "Non-Attentive":  (0, 50, 220),
    "Loading...":     (180, 180, 180),
}

MSG_FRAME = b'F'
MSG_MOOD  = b'M'
MSG_GRID  = b'G'
MSG_ATTN  = b'A'


class AttentiveMobileNetV2(nn.Module):
    """
    MobileNetV2-based binary classifier that predicts whether a person is attentive.
    The ImageNet backbone is frozen at its feature output and a small custom head makes the final call.
    """

    def __init__(self):
        """
        Builds the model by replacing MobileNetV2's default classifier with a lightweight
        two-layer head that outputs a single logit for binary classification.
        """
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Runs a forward pass through the backbone then the classification head.
        Produces a raw logit - apply sigmoid externally to get a probability.

        Args:
            x: A (B, 3, H, W) float tensor of normalised RGB images.

        Returns:
            A (B, 1) tensor of raw logits.
        """
        x = self.backbone(x)
        x = self.head(x)
        return x


def load_model(path: str):
    """
    Loads a saved checkpoint into an AttentiveMobileNetV2 and sets it to eval mode.
    Accepts checkpoints saved as either a plain state dict or a dict with a 'model_state_dict' key.

    Args:
        path: File path to the .pth checkpoint file on disk.

    Returns:
        An AttentiveMobileNetV2 instance in eval mode with the checkpoint weights loaded.
    """
    net  = AttentiveMobileNetV2()
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"Checkpoint loaded: {path}")
    else:
        state = ckpt
    net.load_state_dict(state)
    net.eval()
    return net


_eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_face_detect_lock = threading.Lock()


def _safe_detect(gray: np.ndarray, **kwargs):
    """
    Wraps cv2.CascadeClassifier.detectMultiScale with guard checks to prevent crashes.
    Validates that the input is a non-empty 2D array before attempting detection.

    Args:
        gray: A 2D uint8 numpy array (greyscale image) to run detection on.
        **kwargs: Any keyword arguments accepted by detectMultiScale
                  (e.g. scaleFactor, minNeighbors, minSize).

    Returns:
        A list/array of (x, y, w, h) bounding boxes, or an empty tuple on failure.
    """
    if gray is None or gray.size == 0 or gray.ndim != 2:
        return ()
    if gray.shape[0] < 24 or gray.shape[1] < 24:
        return ()
    if _face_cascade.empty():
        return ()
    try:
        with _face_detect_lock:
            return _face_cascade.detectMultiScale(
                np.ascontiguousarray(gray), **kwargs
            )
    except cv2.error:
        return ()


def _detect_face(frame_bgr: np.ndarray):
    """
    Detects the largest face in a BGR frame using Haar cascades with three fallback passes.
    Each pass loosens detection thresholds and applies image enhancement to handle dim or noisy webcam input.

    Args:
        frame_bgr: A BGR numpy array from the webcam capture.

    Returns:
        A (x, y, w, h) tuple for the largest detected face, or None if no face was found.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = _safe_detect(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    if len(faces):
        return max(faces, key=lambda b: b[2] * b[3])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    faces = _safe_detect(eq, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
    if len(faces):
        return max(faces, key=lambda b: b[2] * b[3])

    den   = cv2.GaussianBlur(eq, (3, 3), 0)
    faces = _safe_detect(den, scaleFactor=1.2, minNeighbors=3, minSize=(36, 36))
    if len(faces):
        return max(faces, key=lambda b: b[2] * b[3])

    return None


_infer_model : AttentiveMobileNetV2 = None
_infer_lock  = threading.Lock()


def run_local_inference(frame_bgr: np.ndarray):
    """
    Detects a face in the frame, crops it, and runs it through the local attentiveness model.
    Returns a Non-Attentive result immediately if no model is loaded or no face is found.

    Args:
        frame_bgr: A BGR numpy array captured from the webcam.

    Returns:
        A tuple of (label, color, score) where label is "Attentive" or "Non-Attentive",
        color is the matching BGR tuple, and score is the sigmoid probability (0.0–1.0).
    """
    if _infer_model is None:
        return "Loading...", MOOD_COLORS["Loading..."], 0.0

    face_box = _detect_face(frame_bgr)
    if face_box is None:
        return "Non-Attentive", MOOD_COLORS["Non-Attentive"], 0.0

    x, y, w, h = face_box
    fh, fw     = frame_bgr.shape[:2]
    x1, y1     = max(0, x), max(0, y)
    x2, y2     = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return "Non-Attentive", MOOD_COLORS["Non-Attentive"], 0.0

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    tensor = _eval_transform(pil_img).unsqueeze(0)

    with _infer_lock:
        with torch.no_grad():
            logit = _infer_model(tensor)
            score = torch.sigmoid(logit).item()

    label = "Attentive" if score >= 0.5 else "Non-Attentive"
    color = MOOD_COLORS.get(label, (180, 180, 180))
    return label, color, score


def run_batch_inference(frames: list) -> tuple:
    """
    Runs inference on a list of frames and averages the scores to smooth out noisy predictions.
    If more than half the frames have no detected face, the result is Non-Attentive.

    Args:
        frames: A list of BGR numpy arrays, typically the contents of the rolling frame buffer.

    Returns:
        A tuple of (label, color, avg_score) based on the average of all valid per-frame scores.
    """
    if not frames:
        return "Loading...", MOOD_COLORS["Loading..."], 0.0

    scores = []
    no_face_count = 0

    for frame in frames:
        _, _, score = run_local_inference(frame)
        if score == 0.0:
            no_face_count += 1
        scores.append(score)

    if no_face_count > len(frames) / 2:
        return "Non-Attentive", MOOD_COLORS["Non-Attentive"], 0.0

    valid_scores = [s for s in scores if s > 0]
    if not valid_scores:
        return "Non-Attentive", MOOD_COLORS["Non-Attentive"], 0.0

    avg_score = np.mean(valid_scores)
    label = "Attentive" if avg_score >= 0.5 else "Non-Attentive"
    color = MOOD_COLORS.get(label, (180, 180, 180))
    return label, color, avg_score


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """
    Reads exactly n bytes from a socket, blocking and looping until all bytes arrive.
    Raises immediately if the remote end closes the connection mid-read.

    Args:
        conn: An active socket connection to read from.
        n: The exact number of bytes expected.

    Returns:
        A bytes object of length n.

    Raises:
        ConnectionError: If the socket closes before all n bytes are received.
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
    Reads one typed message from the socket by first parsing its 5-byte header.
    The header encodes the message type (1 byte) and payload length (4 bytes, big-endian).

    Args:
        conn: An active socket connection to read from.

    Returns:
        A tuple of (message_type, payload) where message_type is a single byte
        and payload contains the raw message body (may be empty bytes).
    """
    hdr    = _recv_exact(conn, 5)
    mtype  = hdr[0:1]
    length = struct.unpack(">I", hdr[1:])[0]
    data   = _recv_exact(conn, length) if length else b""
    return mtype, data


def send_typed(conn: socket.socket, mtype: bytes, data: bytes) -> None:
    """
    Sends a typed message over a socket with a 5-byte framing header prepended.
    Combines the type byte, big-endian length, and payload into one sendall call.

    Args:
        conn: An active socket connection to write to.
        mtype: A single byte identifying the message type (e.g. MSG_FRAME, MSG_ATTN).
        data: The raw bytes payload to transmit.

    Returns:
        None
    """
    conn.sendall(mtype + struct.pack(">I", len(data)) + data)


_latest_frame = None
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()

_mood_label  = "Loading..."
_mood_score  = 0.0
_mood_color  = MOOD_COLORS["Loading..."]
_mood_lock   = threading.Lock()

_frame_buffer      = deque(maxlen=BATCH_SIZE)
_frame_buffer_lock = threading.Lock()

_grid_jpeg   : bytes = b""
_grid_lock   = threading.Lock()

_stop        = threading.Event()
_send_lock   = threading.Lock()


def _open_camera(index: int):
    """
    Tries to open the webcam at the given index, preferring DirectShow on Windows if available.
    Falls back to OpenCV's default backend if the preferred one fails to open.

    Args:
        index: The integer index of the camera device to open (usually 0 for the built-in webcam).

    Returns:
        An opened cv2.VideoCapture object, or None if all backends failed.
    """
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append((cv2.CAP_DSHOW, "DirectShow"))
    backends.append((cv2.CAP_ANY, "Any"))
    for backend, name in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            print(f"[OK] Webcam opened with {name} backend.")
            return cap
        cap.release()
    return None


def _sender(conn: socket.socket):
    """
    Background thread that reads the latest webcam frame, runs batch inference every INFER_EVERY_N
    frames, and continuously streams JPEG-encoded frames to the server at SEND_FPS.
    Sends the averaged attentiveness result over MSG_ATTN whenever the batch buffer is full.

    Args:
        conn: The active socket connection to the server.

    Returns:
        None - runs until _stop is set or a socket error occurs.
    """
    global _mood_label, _mood_score, _mood_color

    interval   = 1.0 / SEND_FPS
    params     = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    frame_ctr  = 0
    print(f"[->] Sender started ({SEND_FPS} fps, batch inference (size={BATCH_SIZE}))")

    try:
        while not _stop.is_set():
            _frame_event.wait(timeout=2.0)
            if not _frame_event.is_set():
                continue
            t = time.monotonic()

            with _frame_lock:
                frame = _latest_frame
                _frame_event.clear()

            if frame is None:
                continue

            if frame_ctr % INFER_EVERY_N == 0:
                with _frame_buffer_lock:
                    _frame_buffer.append(frame.copy())
                    if len(_frame_buffer) == BATCH_SIZE:
                        batch_frames = list(_frame_buffer)
                        label, color, score = run_batch_inference(batch_frames)
                        with _mood_lock:
                            _mood_label = label
                            _mood_score = score
                            _mood_color = color
                        try:
                            with _send_lock:
                                send_typed(conn, MSG_ATTN,
                                           f"{label}:{score:.4f}".encode())
                        except OSError:
                            break

            ok, buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                frame_ctr += 1
                continue
            try:
                with _send_lock:
                    send_typed(conn, MSG_FRAME, buf.tobytes())
            except OSError:
                break

            frame_ctr += 1
            gap = interval - (time.monotonic() - t)
            if gap > 0:
                time.sleep(gap)

    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"[!] Sender error: {e}")
    finally:
        _stop.set()
        print("[->] Sender stopped.")


def _receiver(conn: socket.socket):
    """
    Background thread that listens for incoming messages from the server.
    Stores the latest clean grid JPEG when a MSG_GRID message arrives; ignores MSG_MOOD control messages.

    Args:
        conn: The active socket connection to the server.

    Returns:
        None - runs until _stop is set or a socket error occurs.
    """
    global _grid_jpeg
    print("[<-] Receiver started.")
    try:
        while not _stop.is_set():
            mtype, data = recv_typed(conn)

            if mtype == MSG_MOOD:
                msg = data.decode(errors="replace").strip()
                if msg.startswith("ID:"):
                    continue
            elif mtype == MSG_GRID:
                if data:
                    with _grid_lock:
                        _grid_jpeg = data

    except (ConnectionError, BrokenPipeError, OSError) as e:
        print(f"[!] Receiver error: {e}")
    finally:
        _stop.set()
        print("[<-] Receiver stopped.")


def _draw_own_overlay(grid: np.ndarray, display_label: str,
                      label: str, score: float, color: tuple) -> np.ndarray:
    """
    Stamps a compact self-status panel onto the top-left corner of the grid image.
    Shows the user's name, attentiveness status text, and a coloured border around the whole grid.

    Args:
        grid: The full meeting room grid image received from the server (BGR numpy array).
        display_label: The formatted name+ID string shown as the panel title (e.g. "Alice (C01)").
        label: The current attentiveness label string (e.g. "Attentive", "Non-Attentive").
        score: The float confidence score from the model, used for informational display.
        color: A (B, G, R) tuple that drives the border and label colour.

    Returns:
        A copy of the grid with the overlay panel and border applied.
    """
    out     = grid.copy()
    h, w    = out.shape[:2]

    panel_w, panel_h = 280, 72
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    cv2.putText(out, f"You ({display_label})",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1)

    if label == "Attentive":
        bt, bc = "ATTENTIVE", (0, 210, 0)
    elif label == "Loading...":
        bt, bc = "CONNECTING", (200, 200, 0)
    else:
        bt, bc = "NOT ATTENTIVE", (0, 50, 220)

    cv2.putText(out, bt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.56, bc, 2)
    cv2.putText(out, label, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 3)
    return out


def main():
    """
    Entry point - loads the model, connects to the server, exchanges handshake messages,
    opens the webcam, starts sender/receiver threads, and runs the local display loop.
    Cleans up the webcam and socket gracefully on quit or interrupt.
    """
    global _latest_frame, _infer_model
    if os.path.exists(MODEL_PATH_PTH):
        print(f"[*] Loading checkpoint model: {MODEL_PATH_PTH} ...")
        _infer_model = load_model(MODEL_PATH_PTH)
        print("[OK] Local inference ENABLED (checkpoint)")
    else:
        print(f"[!] No model file found ({MODEL_PATH_PTH})")
        print("    Running WITHOUT inference -- all frames will show 'Non-Attentive'.")

    try:
        server_ip = input("Enter server IP address [10.24.48.12]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Aborted."); sys.exit(0)
    if not server_ip:
        server_ip = "10.24.48.12"

    print(f"\n[*] Connecting to {server_ip}:{PORT}...")
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        conn.connect((server_ip, PORT))
    except Exception as e:
        print(f"[X] Connection failed: {e}"); sys.exit(1)
    print("[OK] Connected!")

    client_id = "??"
    try:
        mtype, data = recv_typed(conn)
        msg = data.decode(errors="replace")
        if msg.startswith("ID:"):
            client_id = msg.split(":", 1)[1]
    except Exception as e:
        print(f"[!] ID handshake error: {e}")
    print(f"[OK] Assigned ID: {client_id}\n")

    try:
        display_name = input("Enter your display name [Guest]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[!] Aborted."); conn.close(); sys.exit(0)
    if not display_name:
        display_name = "Guest"

    try:
        send_typed(conn, MSG_MOOD, f"NAME:{display_name}".encode())
    except Exception as e:
        print(f"[!] Failed to send display name: {e}")

    cap = _open_camera(CAM_INDEX)
    if cap is None or not cap.isOpened():
        print(f"[X] Cannot open webcam index {CAM_INDEX}")
        conn.close(); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Webcam {aw}x{ah}  |  press Q to quit\n")

    threading.Thread(target=_sender,   args=(conn,), daemon=True, name="Sender").start()
    threading.Thread(target=_receiver, args=(conn,), daemon=True, name="Receiver").start()

    win = f"Meeting Room - {display_name} ({client_id})  (Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 520)

    try:
        while not _stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); continue

            with _frame_lock:
                _latest_frame = frame.copy()
                _frame_event.set()

            with _mood_lock:
                lbl, sco, col = _mood_label, _mood_score, _mood_color

            with _grid_lock:
                jpeg = _grid_jpeg

            if jpeg:
                arr  = np.frombuffer(jpeg, dtype=np.uint8)
                grid = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if grid is not None:
                    display = _draw_own_overlay(
                        grid,
                        f"{display_name} ({client_id})",
                        lbl, sco, col
                    )
                    cv2.imshow(win, display)
            else:
                placeholder = frame.copy()
                cv2.putText(placeholder, "Waiting for meeting room...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 0), 2)
                cv2.imshow(win, placeholder)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[*] Quit."); break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        _stop.set()
        cap.release()
        try: conn.shutdown(socket.SHUT_RDWR)
        except Exception: pass
        conn.close()
        cv2.destroyAllWindows()
        print("[OK] Client stopped.")


if __name__ == "__main__":
    main()
