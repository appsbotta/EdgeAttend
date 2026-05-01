# EdgeAttend

> **Real-time attentiveness monitoring for online meetings, powered by edge AI.**

EdgeAttend detects whether each participant in a video call is attentive or not, entirely on the client device, without sending raw video to any cloud service. A MobileNetV2-based binary classifier runs locally on every client machine. A lightweight server aggregates the results, assembles a shared grid view, and streams it back to all participants and to a browser-based monitor.

---

## Table of Contents

- [Demo & Screenshots](#demo--screenshots)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Model](#model)
- [Model Compression Results](#model-compression-results)
- [Prerequisites](#prerequisites)
- [Step-by-Step: Reproduce the Project](#step-by-step-reproduce-the-project)
  - [1 — Install dependencies](#1--install-dependencies)
  - [2 — Download the dataset](#2--download-the-dataset)
  - [3 — Prepare the dataset](#3--prepare-the-dataset)
  - [4 — Train the model](#4--train-the-model)
  - [5 — Compress the model (optional)](#5--compress-the-model-optional)
  - [6 — Evaluate compressed models](#6--evaluate-compressed-models)
  - [7 — Run the live system](#7--run-the-live-system)
- [Configuration Reference](#configuration-reference)
- [References](#references)

---

## Demo & Screenshots

| Server browser monitor | Client window |
|---|---|
| Annotated grid with coloured borders (green = Attentive, red = Non-Attentive) served as MJPEG at `http://<server-ip>:8080` | Clean grid received from server with a self-status overlay in the top-left corner |

Training curves produced by `train.ipynb`:

| Loss | Accuracy | AUC |
|---|---|---|
| ![Loss](plots/loss.png) | ![Accuracy](plots/accuracy.png) | ![AUC](plots/auc.png) |

Compression trade-off graphs produced by the pruning scripts:

| Unstructured pruning | Structural pruning |
|---|---|
| ![Unstructured](plots/unstructured_pruning_tradeoff_graph.png) | ![Structural](plots/struct_pruning_tradeoff_graph.png) |

---

## Project Structure

```
EdgeAttend/
├── client.py                        # Client app — webcam capture, local inference, server streaming
├── server.py                        # Server app — multi-client aggregator, grid composer, MJPEG server
├── requirements.txt                 # Python dependencies
├── report.md                        # Full project report
├── plots/                           # Training and compression graphs
│   ├── accuracy.png
│   ├── auc.png
│   ├── loss.png
│   ├── struct_pruning_tradeoff_graph.png
│   └── unstructured_pruning_tradeoff_graph.png
│
├── Training/                        # Data preparation and model training
│   ├── README.md                    # Training-specific instructions
│   ├── prepare_dataset.py           # Extracts face crops from DAiSEE videos
│   └── train.ipynb                  # Two-stage training notebook
│
└── Edge_Optimization/               # Model compression pipeline
    ├── README.md                    # Compression-specific instructions
    ├── labels.json                  # Class index → label mapping
    ├── quantize_model.py            # Post-training static INT8 quantization (FX graph mode)
    ├── prune_model.py               # Unstructured (L1) pruning with optional fine-tuning
    ├── struct_prune_model.py        # Structural (channel) pruning with optional fine-tuning
    └── evaluate_model.py            # Unified benchmark — accuracy, speed, size for all variants
```

**Generated artefacts** (not committed, produced during reproduction):

```
EdgeAttend/
├── attentive_model.pth                    # Trained FP32 model (output of train.ipynb)
├── dataset_splits.json                    # Train/val/test file-path lists (output of train.ipynb)
├── dataset/                               # Preprocessed image dataset (output of prepare_dataset.py)
│   ├── attentive/
│   └── not_attentive/
└── Edge_Optimization/
    ├── attentive_model_quantized.pth      # INT8 TorchScript model
    ├── best_unstructured_pruned_no_ft_*.pth
    ├── best_unstructured_pruned_ft_*.pth
    ├── best_struct_pruned_no_ft_*.pth
    ├── best_struct_pruned_ft_*.pth
    ├── evaluation_results.json
    └── evaluation_results.log
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT MACHINE                          │
│                                                                  │
│  Webcam → [Frame capture] → [Face detection (Haar cascade)]     │
│               ↓                                                  │
│         [AttentiveMobileNetV2]  ← attentive_model.pth           │
│         (local inference, batch of 5 frames)                    │
│               ↓                                                  │
│         Label + Score  ──MSG_ATTN──►  SERVER (port 9999)        │
│         JPEG frames    ──MSG_FRAME──► SERVER (port 9999)        │
│                                                                  │
│         ◄──MSG_GRID── Clean grid JPEG (no mood annotations)     │
│         [Draw own overlay on top-left] → cv2.imshow             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          SERVER MACHINE                          │
│                                                                  │
│  TCP socket (port 9999) ← accepts multiple clients              │
│  One thread per client (ClientHandler)                           │
│               ↓                                                  │
│  Aggregates frames + attentiveness labels                        │
│               ↓                                                  │
│  [Grid encoder loop @ 8 fps]                                     │
│   ├─ Annotated grid  → MJPEG HTTP (port 8080) / browser        │
│   └─ Clean grid      → MSG_GRID pushed to all clients           │
│                                                                  │
│  GET /         → HTML monitor page                               │
│  GET /stream   → MJPEG stream (annotated)                        │
│  GET /status   → JSON snapshot of all client states             │
└─────────────────────────────────────────────────────────────────┘
```

### Socket message protocol

| Type byte | Constant | Direction | Payload |
|---|---|---|---|
| `F` | `MSG_FRAME` | client → server | JPEG-encoded video frame |
| `M` | `MSG_MOOD` | bidirectional | UTF-8 control strings (`ID:<id>`, `NAME:<name>`) |
| `G` | `MSG_GRID` | server → client | JPEG-encoded clean grid |
| `A` | `MSG_ATTN` | client → server | `<label>:<score>` string |

Each message is framed with a **5-byte header**: 1 byte type + 4 bytes big-endian payload length.

---

## Model

### Architecture — AttentiveMobileNetV2

```
AttentiveMobileNetV2
├── Backbone: MobileNetV2 (ImageNet-pretrained, frozen in Stage 1)
│   ├── Features: 2,225,088 parameters
│   └── Classifier: replaced with nn.Identity()
└── Custom head
    ├── Dropout(0.35)
    ├── Linear(1280 → 64)
    ├── ReLU(inplace=True)
    ├── Dropout(0.25)
    └── Linear(64 → 1)   ← raw logit; apply sigmoid for probability

Total parameters: ~2.4 M
```

Binary classification: **Attentive** (score ≥ 0.5) vs **Non-Attentive** (score < 0.5).

### Dataset — DAiSEE

| Property | Value |
|---|---|
| Source | [IIT Hyderabad](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) |
| Label mapping | Engagement ≥ 2 → **Attentive**, Engagement < 2 → **Non-Attentive** |
| Face extraction | OpenCV Haar cascade, 5 FPS sampling, largest face per frame, 160 × 160 px |
| Target per class | 5 000 images (augmented if needed) |
| Total samples | 10 000 (balanced) |
| Split | 70 % train / 15 % val / 15 % test |

### Training results

| Metric | Value |
|---|---|
| Test Accuracy | **98.27 %** |
| Test Precision | 98.32 % |
| Test Recall | 98.27 % |
| ROC-AUC | 0.9983 |

---

## Model Compression Results

All models evaluated on CPU to reflect edge-device conditions.

| Model variant | Accuracy | Inference (ms/img) | Size (MB) | Parameters |
|---|---:|---:|---:|---:|
| Original FP32 | 98.53 % | 22.68 | 9.02 | 2 305 921 |
| **Quantized INT8** | **98.47 %** | **5.87** | **2.57** | 2 305 921 |
| Unstructured pruned, no fine-tuning | 78.53 % | — | 9.04 | 2 305 921 |
| Unstructured pruned, with fine-tuning | 98.40 % | — | 9.04 | 2 305 921 |
| Structurally pruned, no fine-tuning | 50.00 % | — | 0.27 | 30 477 |
| Structurally pruned, with fine-tuning | 97.47 % | — | 0.27 | 30 477 |

**Best overall for edge deployment**: INT8 quantization — 3.5× smaller, ~4× faster, < 0.1 pp accuracy drop.

---

## Prerequisites

- Python 3.9 or later
- A webcam on each client machine
- The server and all clients must be on the **same network** (or the server port 9999 must be reachable)
- GPU is optional but recommended for training; inference runs on CPU

---

## Step-by-Step: Reproduce the Project

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> `torch-pruning` requires PyTorch ≥ 2.0.  
> Install CUDA-enabled PyTorch first if you want GPU training:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2 — Download the dataset

1. Go to <https://people.iith.ac.in/vineethnb/resources/daisee/index.html> and request access to the **DAiSEE** dataset.
2. Download and extract it so the directory layout matches:

```
Training/
└── DAiSEE/
    ├── DataSet/
    │   └── Train/
    │       └── {person_id}/
    │           └── {clip_id}/
    │               └── {video_file}.avi
    └── Labels/
        └── TrainLabels.csv
```

### 3 — Prepare the dataset

```bash
cd Training
python prepare_dataset.py
```

This script:
- Reads `DAiSEE/Labels/TrainLabels.csv` and maps each clip to an engagement label.
- Samples frames at 5 FPS, detects the largest face with an OpenCV Haar cascade, and saves 160 × 160 crops.
- Balances both classes to 5 000 images each using random augmentation (horizontal flip, brightness, blur).

Output:

```
Training/dataset/
├── attentive/      # 5 000 images  (A_0.jpg … A_N.jpg + aug_*.jpg)
└── not_attentive/  # 5 000 images  (N_0.jpg … N_N.jpg + aug_*.jpg)
```

### 4 — Train the model

Open and run **`Training/train.ipynb`** sequentially.

The notebook performs a two-stage transfer learning strategy:

| Stage | Epochs | Learning rate | Trainable layers |
|---|---|---|---|
| 1 — Head only | 10 | 1 × 10⁻³ | Custom head only |
| 2 — Fine-tuning | 10 | 1 × 10⁻⁵ | Last 40 backbone blocks + head |

Outputs saved to the **`Training/`** folder:
- `attentive_model.pth` — best checkpoint
- `dataset_splits.json` — train/val/test file-path lists (required by compression scripts)

Copy both files to the **project root** and to `Edge_Optimization/` before the next steps:

```bash
cp Training/attentive_model.pth .
cp Training/dataset_splits.json .
cp Training/attentive_model.pth Edge_Optimization/
cp Training/dataset_splits.json Edge_Optimization/
```

### 5 — Compress the model (optional)

All scripts in `Edge_Optimization/` expect `attentive_model.pth` and `dataset_splits.json` in the **same directory** as the script.

#### 5a — Post-training static quantization

```bash
cd Edge_Optimization
python quantize_model.py
```

Converts the FP32 model to INT8 using PyTorch FX graph mode with the `qnnpack` backend.  
Output: `attentive_model_quantized.pth`

#### 5b — Unstructured (L1) pruning

```bash
python prune_model.py
```

Tests pruning ratios from 10 % to 90 % and evaluates accuracy with and without 3-epoch fine-tuning.  
Saves the two best trade-off models, e.g.:
- `best_unstructured_pruned_no_ft_90.pth`
- `best_unstructured_pruned_ft_90.pth`

Also saves `plots/unstructured_pruning_tradeoff_graph.png`.

#### 5c — Structural (channel) pruning

```bash
python struct_prune_model.py
```

Physically removes channels using `torch-pruning` (MagnitudePruner) at ratios 10 %–90 %, with and without fine-tuning.  
Saves the two best trade-off models, e.g.:
- `best_struct_pruned_no_ft_90.pth`
- `best_struct_pruned_ft_90.pth`

Also saves `plots/struct_pruning_tradeoff_graph.png`.

### 6 — Evaluate compressed models

```bash
cd Edge_Optimization
python evaluate_model.py
```

Loads all six model variants, evaluates accuracy on the held-out validation split, measures single-image inference latency (CPU), and records model size.

Outputs:
- `evaluation_results.json` — machine-readable metrics table
- `evaluation_results.log` — timestamped log

### 7 — Run the live system

#### 7a — Start the server

Run on the central machine (can also be one of the participant machines):

```bash
python server.py
```

The server:
- Listens for client connections on **port 9999** (TCP).
- Captures the host webcam and adds it as the first tile in the grid.
- Serves a browser-based annotated monitor at **`http://<server-ip>:8080`**.
- Pushes a clean grid back to every connected client.

Open `http://<server-ip>:8080` in any browser to view the annotated attentiveness grid.  
Open `http://<server-ip>:8080/status` for a live JSON snapshot of all client states.

#### 7b — Start each client

Run on each participant's machine. The model file must be present:

```bash
# Copy attentive_model.pth to the project root on each client machine, then:
python client.py
```

When prompted, enter the server's IP address (press Enter to use the default).

```
Enter server IP address [10.24.48.12]: 192.168.1.42
```

The client:
1. Loads `attentive_model.pth` for local inference.
2. Connects to the server and receives a client ID (e.g., `C01`).
3. Opens the webcam and starts streaming JPEG frames.
4. Runs batch inference (window of 5 frames) every 5 captured frames.
5. Sends attentiveness label + confidence score to the server.
6. Receives the clean grid from the server and displays it locally with a self-status overlay.

Press **Q** in the client window to disconnect.

---

## Configuration Reference

### client.py

| Constant | Default | Description |
|---|---|---|
| `PORT` | `9999` | Server TCP port |
| `SEND_FPS` | `15` | Frames sent to server per second |
| `JPEG_QUALITY` | `100` | JPEG encode quality for transmitted frames |
| `CAM_INDEX` | `0` | Webcam device index |
| `CAM_W / CAM_H` | `640 / 480` | Webcam capture resolution |
| `MODEL_PATH_PTH` | `attentive_model.pth` | Path to the .pth checkpoint |
| `IMG_SIZE` | `160` | Face crop size fed to the model |
| `INFER_EVERY_N` | `5` | Run inference once every N captured frames |
| `BATCH_SIZE` | `5` | Rolling window size for batch inference |

### server.py

| Constant | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `9999` | TCP socket port for clients |
| `MJPEG_PORT` | `8080` | HTTP port for browser monitor |
| `TILE_W / TILE_H` | `400 / 300` | Pixel dimensions of each participant tile |
| `MAX_COLS` | `3` | Maximum tiles per row in the grid |
| `GRID_FPS` | `8` | Grid rebuild / push rate |

---

## References

1. A. Gupta, A. DCunha, K. Awasthi, V. Balasubramanian, *DAiSEE: Towards User Engagement Recognition in the Wild*, arXiv:1609.01885 (ICMI 2022). <https://arxiv.org/abs/1609.01885>
2. M. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
3. PyTorch quantization documentation: <https://pytorch.org/docs/stable/quantization.html>
4. Torch-Pruning library: <https://github.com/VainF/Torch-Pruning>
