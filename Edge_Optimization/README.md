# Model Compression Pipeline

This pipeline covers **Post-Training Static Quantization**, **Unstructured (L1) Pruning**, and **Structural (Channel) Pruning**, along with a unified evaluation script to benchmark all compressed variants head-to-head.

## Usage Instructions

#### Step - 1: 
Change the location of `dataset_splits.json` and `attentive_model.pth` to this folder

#### Step - 2: Quantization
```bash
python quantize_model.py
```

#### Step - 3: Structured Prunning
```bash
python struct_prune_model.py
```

#### Step - 4: Unstructured Prunning
```bash
python prune_model.py
```

#### Step - 5: Evaluation
```bash
python evaluate_model.py
```
