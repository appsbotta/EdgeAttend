# Data Preprocessing Pipeline

### Overview
The data preprocessing (`prepare_dataset.py`) converts raw video data from the DAiSEE dataset into a balanced, augmented image dataset ready for model training.

### Dataset Source
- **Source**: DAiSEE (Dataset for Engagement Estimation in Education)
- **Directory Structure**:
  ```
  DAiSEE/
  ├── DataSet/
  │   └── Train/
  │       └── {person_id}/
  │           └── {clip_id}/
  │               └── {video_file}.avi
  └── Labels/
      └── TrainLabels.csv
  ```

#### Step 1: Label Loading
#### Step 2: Face Detection
#### Step 3: Dataset Organization
#### Step 4: Data Augmentation & Balancing
If a class has fewer than 5,000 samples, augmentation fills the gap

Output structure after processing:
```
dataset/
├── attentive/          (engagement ≥ 2)
│   ├── A_0.jpg
│   ├── A_1.jpg
│   └── ...
└── not_attentive/      (engagement < 2)
    ├── N_0.jpg
    ├── N_1.jpg
    └── ...
```

# Model Training Pipeline

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 160 | Input image dimension (160×160 pixels) |
| `BATCH_SIZE` | 32 | Mini-batch size for training |
| `NUM_WORKERS` | 4 | Parallel data loading workers |
| `VAL_RATIO` | 0.15 | 15% of data for validation |
| `TEST_RATIO` | 0.15 | 15% of data for testing |
| `TRAIN_RATIO` | 0.70 | 70% of data for training |
| `INITIAL_EPOCHS` | 10 | Stage 1 training epochs |
| `FINE_TUNE_EPOCHS` | 10 | Stage 2 fine-tuning epochs |
| `FINE_TUNE_LAYERS` | 40 | Number of backbone blocks to unfreeze |

### Model Architecture: AttentiveMobileNetV2

```
AttentiveMobileNetV2
├── Backbone: MobileNetV2 (ImageNet pretrained)
│   ├── Features (frozen initially): 2,225,088 parameters
│   └── Classifier: Replaced with Identity layer
└── Custom Head
    ├── Dropout(0.35)
    ├── Linear(1280 → 64)
    ├── ReLU(inplace=True)
    ├── Dropout(0.25)
    └── Linear(64 → 1)  [Binary output, logits]
```

## Training Pipeline

#### Stage 1: Head-Only Training (10 epochs)
#### Stage 2: Fine-Tuning (10 epochs)
---

## Usage Instructions

#### Step - 1: Download the dataset from IIT Hyderabad
https://people.iith.ac.in/vineethnb/resources/daisee/index.html

#### Step - 2: Place the dataset in Training Folder

#### Step - 3: Data Preparation
Run the Following command in the terminal to prepare the dataset for training:
```bash
python prepare_dataset.py
```
**Prerequisites**: 
- DAiSEE dataset directory structure in place

**Output**: Balanced dataset in `dataset/` directory

### Step 2: Model Training
Open and run `train.ipynb` sequentially



## References
- **DAiSEE Dataset**: [Palmar Gupta et al., "DAiSEE: A Large-scale Dataset for Engagement Estimation in Education" (ICMI 2022)](https://arxiv.org/abs/1609.01885)

