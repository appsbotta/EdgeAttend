# AttentiveMobileNetV2 — Model Compression Pipeline

A complete pipeline for training, compressing, and evaluating a binary attention classifier built on MobileNetV2. The pipeline covers **Post-Training Static Quantization**, **Unstructured (L1) Pruning**, and **Structural (Channel) Pruning**, along with a unified evaluation script to benchmark all compressed variants head-to-head.

## Usage Instructions

#### Step - 1: Change the dataset_splits.json and attentive_model.pth file paths

#### Step - 2: Quantization
Open and run `quantize_model.py`

#### Step - 2: Structured Prunning
Open and run `struct_prune_model.py`

#### Step - 2: Unstructured Prunning
Open and run `prune_model.py`

#### Step - 2: Evaluation
Open and run `evaluate_model.py`




