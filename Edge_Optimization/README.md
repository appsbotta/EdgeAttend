# Model Compression Pipeline (Edge Optimization)

This pipeline covers **Post-Training Static Quantization**, **Unstructured Pruning**, and **Structural Pruning**, along with a unified evaluation script to benchmark all compressed variants.

---

##  Update Paths

Before running any script, open each file and update the dataset and model weight paths to match your local setup. Also modify the dataset_splits.json file for exact file paths or run the prepare dataset file in this folder.

### `prune_model.py`
```python
# Line ~180 — path to your dataset splits JSON
train_loader, val_loader = get_dataloaders(splits_path="dataset_splits.json")

# Line ~221 — path to your baseline model weights
checkpoint = torch.load("attentive_model.pth", map_location=device)
```

### `quantize_model.py`
```python
# Line ~100 — path to your dataset splits JSON
with open(splits_path, "r") as f:   # default: "dataset_splits.json"

# Line ~115 — path to your baseline model weights
checkpoint = torch.load("attentive_model.pth", map_location=device)
```

### `struct_prune_model.py`
```python
# Line ~160 — path to your dataset splits JSON
train_loader, val_loader = get_dataloaders(splits_path="dataset_splits.json")

# Line ~133 — path to your baseline model weights (inside load_base_model)
checkpoint = torch.load(model_path, map_location=device)   # default: "attentive_model.pth"
```

### `evaluate_model.py`
```python
# Line ~55 — path to your dataset splits JSON
val_loader = get_val_dataloader(splits_path="dataset_splits.json")

# Lines ~110–117 — paths to each model file you want to benchmark
models_to_eval = [
    ("Original FP32 Model",            "attentive_model.pth"),
    ("Quantized INT8 Model",            "attentive_model_quantized.pth"),
    ("Unstructured Pruned (No FT)",     "best_unstructured_pruned_no_ft_90.pth"),
    ...
]
```

---


## Installation

```bash
pip install torch torchvision scikit-learn tqdm pillow matplotlib torch-pruning
```

> **Note:** `torch-pruning` is required only for `struct_prune_model.py`.

---

## Execution Order

Run the scripts in the following order. Each step depends on outputs from the previous one.

---

### Step 1 — Unstructured Pruning &nbsp;`prune_model.py`

Performs an L1-unstructured pruning grid search across sparsity ratios from 10% to 90%. For each ratio it evaluates accuracy both **without** and **with** 3-epoch fine-tuning, then picks the best accuracy–sparsity tradeoff and saves that model.

```bash
python prune_model.py
```

**Outputs:**
- `best_unstructured_pruned_no_ft_<ratio>.pth` — best sparse model without fine-tuning
- `best_unstructured_pruned_ft_<ratio>.pth` — best sparse model with fine-tuning
- `unstructured_pruning_tradeoff_graph.png` — accuracy vs. sparsity plot

---

### Step 2 — INT8 Static Quantization &nbsp;`quantize_model.py`

Applies FX-Graph-based post-training static quantization using the `qnnpack` backend. A small calibration subset from the training split is used to compute activation ranges before conversion to INT8.

```bash
python quantize_model.py
```

**Outputs:**
- `attentive_model_quantized.pth` — TorchScript INT8 model (~4× smaller than FP32)

---

### Step 3 — Structural (Channel) Pruning &nbsp;`struct_prune_model.py`

Uses `torch-pruning` (`MagnitudePruner`) to physically remove entire channels/filters from the network at ratios from 10% to 90%. Unlike unstructured pruning, the model architecture itself shrinks, giving real reductions in both parameter count and inference FLOPs.

```bash
python struct_prune_model.py
```

**Outputs:**
- `best_struct_pruned_no_ft_<ratio>.pth` — best structurally pruned model without fine-tuning
- `best_struct_pruned_ft_<ratio>.pth` — best structurally pruned model with fine-tuning
- `struct_pruning_tradeoff_graph.png` — accuracy vs. channel sparsity plot

---

### Step 4 — Unified Evaluation &nbsp;`evaluate_model.py`

Loads all six model variants (original, quantized, two unstructured pruned, two structurally pruned) and runs them through the held-out validation set. Reports accuracy, precision, recall, F1, inference time, model size, parameter count, and weight sparsity for every variant.

> Evaluation is forced to **CPU** to get accurate INT8 and sparse speedup measurements.

```bash
python evaluate_model.py
```

**Outputs:**
- `evaluation_results.json` — all metrics in machine-readable form
- `evaluation_results.log` — timestamped log of every run

---




## Dataset Split Format

`dataset_splits.json` must contain absolute (or relative) file paths split into training and validation sets:

```json
{
  "train_files": [
    "/path/to/data/attentive/img001.jpg",
    "/path/to/data/not_attentive/img002.jpg"
  ],
  "val_files": [
    "/path/to/data/attentive/img101.jpg",
    "/path/to/data/not_attentive/img102.jpg"
  ]
}
```

Labels are inferred from the file path — any path containing the substring `not_attentive` is assigned label `0`; all others are assigned label `1`.
