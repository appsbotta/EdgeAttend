import os
import json
import logging
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image

# --- Setup Logging ---
logging.basicConfig(
    filename='evaluation_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Original Model Definition ---
class AttentiveMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
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
        x = self.backbone(x)
        x = self.head(x)
        return x

# --- Dataloader Definition ---
class ValidationPathDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.paths = file_paths
        self.transform = transform
        # Strictly enforce the mapping to prevent label inversion
        self.label_mapping = {'not_attentive': 0, 'attentive': 1}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        class_name = "attentive" if "not_attentive" not in path else "not_attentive"
        label = torch.tensor(self.label_mapping[class_name], dtype=torch.float32)
        
        return image, label

def get_val_dataloader(splits_path="dataset_splits.json", img_size=160, batch_size=32):
    """Loads the validation subset to prevent data leakage."""
    try:
        with open(splits_path, "r") as f:
            splits = json.load(f)
        val_files = splits.get("val_files", splits.get("valid_files", []))
        if not val_files:
            raise ValueError(f"Could not find 'val_files' in {splits_path}")
    except FileNotFoundError:
        print(f"Error: {splits_path} not found. You must save your train/val split from train.ipynb.")
        return None

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = ValidationPathDataset(val_files, transform=eval_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Measurement Helpers ---
def measure_sparsity(model):
    """Calculates the percentage of individual zeroed weights (Unstructured Pruning)."""
    zeros = 0
    elements = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            zeros += torch.sum(module.weight == 0).item()
            elements += module.weight.nelement()
    return (zeros / elements) * 100.0 if elements > 0 else 0.0

def count_parameters(model):
    """Calculates total parameter count (Structured Pruning)."""
    return sum(p.numel() for p in model.parameters())

# --- Core Evaluation Function ---
def evaluate_model(model, data_loader, device, model_name, model_path):
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model.to(device)
    
    all_labels = []
    all_preds = []
    
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc=f"Evaluating {model_name}")
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            logits = model(images)
            
            # Handle TorchScript tuples if present
            if isinstance(logits, tuple):
                logits = logits[0]
                
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            
            all_labels.extend(labels.numpy().ravel())
            all_preds.extend(preds)
            
    inference_time = time.time() - start_time
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
    sparsity = measure_sparsity(model)
    total_params = count_parameters(model)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "inference_time_seconds": float(inference_time),
        "model_size_mb": float(file_size_mb),
        "total_parameters": int(total_params),
        "sparsity_percent": float(sparsity)
    }
    
    logging.info(f"Results for {model_name}: {metrics}")
    print(f"  Acc: {acc:.4f} | Size: {file_size_mb:.2f} MB | Params: {total_params:,} | Sparsity: {sparsity:.2f}%")
    return metrics

def main():
    device = torch.device("cpu") 
    print(f"Evaluating on {device} (CPU forced to accurately measure INT8 and sparse speedups)...")
    
    val_loader = get_val_dataloader()
    if val_loader is None:
        return

    results = {}
    
    # Define all models to run through the gauntlet
    models_to_eval = [
        ("Original FP32 Model", "attentive_model.pth"),
        ("Quantized INT8 Model", "attentive_model_quantized.pth"),
        ("Unstructured Pruned (No FT)", "best_unstructured_pruned_no_ft_90.pth"),
        ("Unstructured Pruned (With FT)", "best_unstructured_pruned_ft_90.pth"),
        ("Structurally Pruned (No FT)", "best_struct_pruned_no_ft_90.pth"),
        ("Structurally Pruned (With FT)", "best_struct_pruned_ft_90.pth")
    ]

    for name, path in models_to_eval:
        print(f"\n--- Loading {name} ---")
        if not os.path.exists(path):
            print(f"File {path} not found. Skipping...")
            continue
            
        # weights_only=False allows loading both state_dicts and full custom model objects
        loaded_data = torch.load(path, map_location=device, weights_only=False)
        
        # SMART LOADING LOGIC
        if isinstance(loaded_data, dict):
            # It's a state_dict, we need to initialize the architecture first
            model = AttentiveMobileNetV2()
            
            if "Quantized" in name:
                # Re-apply dynamic quantization wrappers before loading INT8 weights
                model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                
            model.load_state_dict(loaded_data.get("model_state_dict", loaded_data))
        else:
            # It's a full model object (Structurally Pruned)
            model = loaded_data
            
        results[name] = evaluate_model(model, val_loader, device, name, path)

    # Save aggregated results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nUniversal Evaluation Complete!")
    print("All metrics saved to evaluation_results.json and evaluation_results.log")

if __name__ == "__main__":
    main()