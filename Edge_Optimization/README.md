# AttentiveMobileNetV2 — Model Compression Pipeline

A complete pipeline for training, compressing, and evaluating a binary attention classifier built on MobileNetV2. The pipeline covers **Post-Training Static Quantization**, **Unstructured (L1) Pruning**, and **Structural (Channel) Pruning**, along with a unified evaluation script to benchmark all compressed variants head-to-head.

---

##  Before You Begin — Update Paths

Before running any script, open each file and update the dataset and model weight paths to match your local setup. Also modify the file paths of dataset_splits.json and attentive_model.pth files.

### `prune_model.py`
```python
train_loader, val_loader = get_dataloaders(splits_path="dataset_splits.json")

checkpoint = torch.load("attentive_model.pth", map_location=device)
```

### `quantize_model.py`
```python
with open(splits_path, "r") as f:   # default: "dataset_splits.json"

checkpoint = torch.load("attentive_model.pth", map_location=device)
```

### `struct_prune_model.py`
```python
train_loader, val_loader = get_dataloaders(splits_path="dataset_splits.json")

checkpoint = torch.load(model_path, map_location=device)   # default: "attentive_model.pth"
```

### `evaluate_model.py`
```python
val_loader = get_val_dataloader(splits_path="dataset_splits.json")

models_to_eval = [
    ("Original FP32 Model",            "attentive_model.pth"),
    ("Quantized INT8 Model",            "attentive_model_quantized.pth"),
    ("Unstructured Pruned (No FT)",     "best_unstructured_pruned_no_ft_90.pth"),
    ...
]
```

---

## Execution Order

Run the scripts in the following order. Each step depends on outputs from the previous one.

---

### Step 1 — Unstructured Pruning &nbsp;`prune_model.py`

```bash
python prune_model.py
```

### Step 2 — INT8 Static Quantization &nbsp;`quantize_model.py`

```bash
python quantize_model.py
```

### Step 3 — Structural (Channel) Pruning &nbsp;`struct_prune_model.py`

```bash
python struct_prune_model.py
```

### Step 4 — Unified Evaluation &nbsp;`evaluate_model.py`

> Evaluation is forced to **CPU** to get accurate INT8 and sparse speedup measurements.

```bash
python evaluate_model.py
```



