import os
import json
import torch
import torch.nn as nn
import numpy as np
import copy
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# QNNPACK is the standard backend for ARM/Mobile CPUs and general INT8 CPU inference
torch.backends.quantized.engine = 'qnnpack'

# 1. Model Definition

class AttentiveMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 architecture built for binary classification.
    We need the exact same architecture here so PyTorch can load the 
    floating-point weights before we compress them.
    """
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        in_features = backbone.classifier[1].in_features
        
        # Rip out the original classifier and replace it with a pass-through
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

# 2. Dataset and Calibration Loaders

class ImagePathDataset(Dataset):
    """
    A simplified dataset wrapper. 
    Notice we DO NOT return labels here. Why? Because quantization calibration 
    only cares about the physical numbers (activations) flowing through the 
    network. It doesn't care if the prediction is right or wrong right now.
    """
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_calibration_dataloader(splits_path="dataset_splits.json", img_size=160, batch_size=32, num_batches=10):
    """
    Loads a small subset of the training data to calibrate activation ranges.
    
    In Static Quantization, the model needs to "watch" real data pass through 
    so it knows the typical minimum and maximum values of the activations. 
    You don't need the whole dataset—usually, 100 to 500 images are plenty.
    
    Args:
        splits_path (str): Path to your JSON split file.
        img_size (int): Image resolution.
        batch_size (int): Images per batch.
        num_batches (int): How many batches to use (32 * 10 = 320 images).
        
    Returns:
        DataLoader: A PyTorch dataloader yielding only images.
    """
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    # Grab just a slice of the training files
    train_files = splits["train_files"]
    calib_files = train_files[:batch_size * num_batches]

    calib_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImagePathDataset(calib_files, transform=calib_transform)
    # Shuffle=False because order doesn't matter for calibration
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 3. Main Quantization Pipeline

def main():
    print("Loading original model...")

    device = torch.device('cpu') 
    
    model = AttentiveMobileNetV2()
    try:
        checkpoint = torch.load("attentive_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    except FileNotFoundError:
        print("Error: attentive_model.pth not found! Please provide the trained weights.")
        return
        
    model.eval()

    orig_size = os.path.getsize("attentive_model.pth") / (1024 * 1024)
    print(f"Original Model Size: {orig_size:.2f} MB")

    print("Preparing FX Graph and inserting observers...")
    
    model_to_quantize = copy.deepcopy(model)
    
    qconfig_mapping = get_default_qconfig_mapping('qnnpack')
    

    example_inputs = (torch.randn(1, 3, 160, 160),)
    
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)

    print("Calibrating model with sample data...")
    calib_loader = get_calibration_dataloader()
    
    with torch.no_grad():
        for batch_idx, images in enumerate(calib_loader):
            prepared_model(images)
            print(f"  Calibrated batch {batch_idx + 1}/{len(calib_loader)}")

    print("Converting model to INT8...")

    quantized_model = convert_fx(prepared_model)

    print("Tracing and saving TorchScript model...")
 
    traced_quantized_model = torch.jit.trace(quantized_model, example_inputs[0])
    
    save_path = "attentive_model_quantized.pth"
    torch.jit.save(traced_quantized_model, save_path)
    
    quant_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nFinal Quantized Model Size: {quant_size:.2f} MB")
    print(f"Compression Ratio: {orig_size / quant_size:.2f}x (Expect ~4.0x)")
    print(f"Success! Model saved to {save_path}")

if __name__ == "__main__":
    main()