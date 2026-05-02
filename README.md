# EdgeAttend

> **Real-time attentiveness monitoring for online meetings, powered by edge AI.**

EdgeAttend detects whether each participant in a video call is attentive or not, entirely on the client device. A MobileNetV2-based binary classifier runs locally on every client machine. A lightweight server aggregates the results, assembles a shared grid view, and streams it back to all participants and to a browser-based monitor.

---

## Table of Contents
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Step-by-Step: Reproduce the Project](#step-by-step-reproduce-the-project)
  - [1 - Install dependencies](#1---install-dependencies)
  - [2 - Download the dataset](#2---download-the-dataset)
  - [3 - Prepare the dataset](#3---prepare-the-dataset)
  - [4 - Train the model](#4---train-the-model)
  - [5 - Compress the model](#5---compress-the-model)
  - [6 - Evaluate compressed models](#6---evaluate-compressed-models)
  - [7 - Run the live system](#7---run-the-live-system)
## Project Structure

```
EdgeAttend/
├── client.py                        # Client app — webcam capture, local inference, server streaming
├── server.py                        # Server app — multi-client aggregator, grid composer
├── requirements.txt                 # Python dependencies
├── report.md                        # Full project report
├── plots/                           # Training and compression graphs
│   ├── accuracy.png
│   ├── auc.png
│   ├── loss.png
│   ├── struct_pruning_tradeoff_graph.png
│   ├── unstructured_pruning_tradeoff_graph.png
│   ├── all_attentive.png
│   ├── all_non_attentive.png
│   └── one_non_attentive.png
│  
│   
│
├── Training/                        # Data preparation and model training
│   ├── README.md                    # Training-specific instructions
│   ├── prepare_dataset.py           # Extracts face crops from DAiSEE videos
│   └── train.ipynb                  # Two-stage training notebook
│
└── Edge_Optimization/               # Model compression pipeline
    ├── README.md                    # Compression-specific instructions
    ├── labels.json                  # Class index → label mapping
    ├── quantize_model.py            # Post-training static INT8 quantization 
    ├── prune_model.py               # Unstructured (L1) pruning with optional fine-tuning
    ├── struct_prune_model.py        # Structural (channel) pruning with optional fine-tuning
    └── evaluate_model.py            # Unified benchmark — accuracy, speed, size for all variants
```
## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT MACHINE                         │
│                                                                 │
│  Webcam → [Frame capture] → [Face detection (Haar cascade)]     │
│               ↓                                                 │
│         [AttentiveMobileNetV2]  ← attentive_model.pth           │
│         (local inference, batch of 5 frames)                    │
│               ↓                                                 │
│         Label + Score  ──MSG_ATTN──►  SERVER (port 9999)        │
│         JPEG frames    ──MSG_FRAME──► SERVER (port 9999)        │
│                                                                 │
│         ◄──MSG_GRID── JPEG Grid                                 │
│         [Draw own overlay on top-left] → cv2.imshow             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          SERVER MACHINE                         │
│                                                                 │
│  TCP socket (port 9999) ← accepts multiple clients              │
│  One thread per client (ClientHandler)                          │
│               ↓                                                 │
│  Aggregates frames + attentiveness labels                       │
│               ↓                                                 │
│  [Grid encoder loop]                                            │
│   ├─ Annotated grid  → HTTP                                     │
│   └─ Clean grid      → MSG_GRID pushed to all clients           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.9 or later
- A webcam on each client machine and server machine
- The server and all clients must be on the **same network**
- GPU is optional but recommended for training, inference runs on CPU

---

## Step-by-Step: Reproduce the Project

### 1 - Install dependencies

```bash
pip install -r requirements.txt
```

> `torch-pruning` requires PyTorch ≥ 2.0.  
> Install CUDA-enabled PyTorch first if you want GPU training:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2 - Download the dataset

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

### 3 - Prepare the dataset

```bash
cd Training
python prepare_dataset.py
```
Output:

```
Training/dataset/
├── attentive/      # 5000 images  (A_0.jpg … A_N.jpg + aug_*.jpg)
└── not_attentive/  # 5000 images  (N_0.jpg … N_N.jpg + aug_*.jpg)
```

### 4 - Train the model

Open and run **`Training/train.ipynb`** sequentially.
Outputs saved to the **`Training/`** folder:
- `attentive_model.pth` - best checkpoint
- `dataset_splits.json` - train/val/test file-path lists 

Copy both files to the **project root** and to `Edge_Optimization/` before the next steps:

```bash
cp Training/attentive_model.pth .
cp Training/dataset_splits.json .
cp Training/attentive_model.pth Edge_Optimization/
cp Training/dataset_splits.json Edge_Optimization/
```

### 5 - Compress the model

All scripts in `Edge_Optimization/` expect `attentive_model.pth` and `dataset_splits.json` in the **same directory** as the script.

#### 5a - Post-training static quantization

```bash
cd Edge_Optimization
python quantize_model.py
```

Converts the FP32 model to INT8 using PyTorch FX graph mode with the `qnnpack` backend.  
Output: `attentive_model_quantized.pth`

#### 5b - Unstructured (L1) pruning

```bash
python prune_model.py
```

Tests pruning ratios from 10 % to 90 % and evaluates accuracy with and without 3-epoch fine-tuning.  
Saves the two best trade-off models, e.g.:
- `best_unstructured_pruned_no_ft_90.pth`
- `best_unstructured_pruned_ft_90.pth`

Also saves `plots/unstructured_pruning_tradeoff_graph.png`.

#### 5c - Structural (channel) pruning

```bash
python struct_prune_model.py
```

Removes channels using `torch-pruning` (MagnitudePruner) at ratios 10 %–90 %, with and without fine-tuning.  
Saves the two best trade-off models, e.g.:
- `best_struct_pruned_no_ft_90.pth`
- `best_struct_pruned_ft_90.pth`

Also saves `plots/struct_pruning_tradeoff_graph.png`.

### 6 - Evaluate compressed models

```bash
cd Edge_Optimization
python evaluate_model.py
```

Loads all six model variants, evaluates accuracy on the held-out validation split, measures single-image inference latency (CPU), and records model size.

Outputs:
- `evaluation_results.json` — machine-readable metrics table
- `evaluation_results.log` — timestamped log

### 7 - Run the live system

#### 7a - Start the server

Run on the central machine (can also be one of the participant machines):

```bash
python server.py
```
#### 7b - Start each client

Run on each participant's machine. The model file must be present:

```bash
# Copy attentive_model_quantized.pth to the project root on each client machine, then:
python client.py
```

When prompted, enter the server's IP address (press Enter to use the default).

```
Enter server IP address [10.24.48.12]: 192.168.1.42
```

Press **Q** in the client window to disconnect.

---
