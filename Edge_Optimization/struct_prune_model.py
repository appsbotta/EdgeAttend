import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

import torch_pruning as tp

# 1. Model Definition

class AttentiveMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 architecture built for binary classification.
    Instead of 1000 ImageNet classes, it outputs a single value indicating 
    whether a user is 'attentive' or 'not_attentive'.
    """
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        
        # Capture the size of the tensor coming out of the backbone
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

# 2. Dataset and Dataloaders

class ImagePathDataset(Dataset):
    """
    A simple dataset that reads image paths and assigns labels based on the string.
    """
    def __init__(self, file_paths, transform=None):
        self.paths = file_paths
        self.transform = transform
        self.label_mapping = {'not_attentive': 0, 'attentive': 1}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        class_name = "not_attentive" if "not_attentive" in path else "attentive"
        label = torch.tensor(self.label_mapping[class_name], dtype=torch.float32)
        
        return image, label


def get_dataloaders(splits_path="dataset_splits.json", img_size=160, batch_size=32):
    """
    Reads the data split JSON and creates PyTorch DataLoaders.
    
    Args:
        splits_path (str): Path to JSON containing train/val split lists.
        img_size (int): Resolution to resize images to.
        batch_size (int): Number of images processed at once.
        
    Returns:
        tuple: (train_loader, val_loader) or (None, None) if files are missing.
    """
    try:
        with open(splits_path, "r") as f:
            splits = json.load(f)
        
        train_files = splits.get("train_files", [])
        val_files = splits.get("val_files", splits.get("test_files", [])) 
    except FileNotFoundError:
        print(f"Error: Could not find {splits_path}. Make sure the file exists.")
        return None, None

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImagePathDataset(train_files, transform=train_transform)
    val_dataset = ImagePathDataset(val_files, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# 3. Helper Functions

def evaluate_model(model, dataloader, device):
    """
    Calculates the binary classification accuracy of the model on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Data to test against (usually validation set).
        device (torch.device): 'cuda' or 'cpu'.
        
    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total if total > 0 else 0.0


def finetune_model(model, dataloader, device, epochs=3, lr=5e-5):
    """
    Trains the model briefly to recover accuracy lost after structural pruning.
    
    Args:
        model (nn.Module): The physically pruned model.
        dataloader (DataLoader): Training data.
        device (torch.device): 'cuda' or 'cpu'.
        epochs (int): Number of passes over the dataset.
        lr (float): Learning rate (keep this small for fine-tuning!).
        
    Returns:
        nn.Module: The fine-tuned model.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
    return model


def load_base_model(device, model_path="attentive_model.pth"):
    """
    Instantiates a fresh model and loads the baseline weights into it.
    
    Why is this a separate function?
    Because structural pruning physically alters the architecture. If we don't 
    start fresh for every loop iteration, a 10% prune followed by a 20% prune 
    would compound, ruining our grid search.
    
    Args:
        device (torch.device): 'cuda' or 'cpu'.
        model_path (str): Path to baseline weights.
        
    Returns:
        nn.Module: A fresh, unpruned base model.
    """
    base_model = AttentiveMobileNetV2()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        base_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Using uninitialized weights.")
    base_model.to(device)
    return base_model

# 4. Main Search & Plotting Logic

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    if train_loader is None or len(val_loader.dataset) == 0:
        print("Warning: Missing training or validation data.")
        return

    print("\nLoading original baseline model...")
    base_model = load_base_model(device)
    baseline_acc = evaluate_model(base_model, val_loader, device)
    
    orig_params = sum(p.numel() for p in base_model.parameters())
    print(f"Baseline Validation Accuracy: {baseline_acc:.4f}")
    print(f"Baseline Parameters: {orig_params:,}")

    # 3. Setup Grid Search
    prune_ratios = [i/10.0 for i in range(1, 10)] 
    acc_no_ft_list = []
    acc_ft_list = []
    
    best_tradeoff_no_ft, best_ratio_no_ft, best_model_obj_no_ft = -1.0, 0.0, None
    best_tradeoff_ft, best_ratio_ft, best_model_obj_ft = -1.0, 0.0, None

    example_inputs = torch.randn(1, 3, 160, 160).to(device)
    
    imp = tp.importance.MagnitudeImportance(p=2)
    

    pruning_weight = 0.5 

    for ratio in prune_ratios:
        print(f"\n--- Testing Structural Pruning Ratio: {ratio*100:.0f}% ---")
        
        model = load_base_model(device)
        
       
        ignored_layers = []
        for m in model.head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                ignored_layers.append(m)

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            ch_sparsity=ratio,
            ignored_layers=ignored_layers,
        )

        pruner.step()
        
        pruned_params = sum(p.numel() for p in model.parameters())
        print(f"Physically shrunk to {pruned_params:,} parameters ({(1 - pruned_params/orig_params)*100:.2f}% reduction)")

        acc_no_ft = evaluate_model(model, val_loader, device)
        acc_no_ft_list.append(acc_no_ft)
        print(f"Accuracy (No Fine-tuning): {acc_no_ft:.4f}")
        
        tradeoff_score_no_ft = acc_no_ft + (ratio * pruning_weight)
        if tradeoff_score_no_ft > best_tradeoff_no_ft:
            best_tradeoff_no_ft = tradeoff_score_no_ft
            best_ratio_no_ft = ratio
            # Deepcopy BEFORE fine-tuning mutates the weights
            best_model_obj_no_ft = copy.deepcopy(model) 

        model = finetune_model(model, train_loader, device, epochs=3, lr=5e-5)
        acc_ft = evaluate_model(model, val_loader, device)
        acc_ft_list.append(acc_ft)
        print(f"Accuracy (With Fine-tuning): {acc_ft:.4f}")
        
        tradeoff_score_ft = acc_ft + (ratio * pruning_weight)
        if tradeoff_score_ft > best_tradeoff_ft:
            best_tradeoff_ft = tradeoff_score_ft
            best_ratio_ft = ratio
            best_model_obj_ft = copy.deepcopy(model) 

    print(f"\n=======================================================")
    
    
    save_path_no_ft = f"best_struct_pruned_no_ft_{int(best_ratio_no_ft*100)}.pth"
    torch.save(best_model_obj_no_ft, save_path_no_ft) 
    print(f"Best Tradeoff (No Fine-Tuning) Found at {best_ratio_no_ft*100:.0f}% Sparsity")
    print(f"Saved optimal No-FT model to: {save_path_no_ft}")
    
    print("-------------------------------------------------------")

    save_path_ft = f"best_struct_pruned_ft_{int(best_ratio_ft*100)}.pth"
    torch.save(best_model_obj_ft, save_path_ft) 
    print(f"Best Tradeoff (With Fine-Tuning) Found at {best_ratio_ft*100:.0f}% Sparsity")
    print(f"Saved optimal FT model to: {save_path_ft}")
    print(f"=======================================================")

    plt.figure(figsize=(10, 6))
    
    plt.plot([r*100 for r in prune_ratios], acc_no_ft_list, marker='o', linestyle='dashed', label='Without Fine-Tuning')
    plt.plot([r*100 for r in prune_ratios], acc_ft_list, marker='s', linestyle='-', label='With Fine-Tuning')
    plt.axhline(y=baseline_acc, color='r', linestyle=':', label='Baseline (Original)')
    
    plt.axvline(x=best_ratio_no_ft*100, color='b', linestyle='-.', alpha=0.5, label='Best No-FT Tradeoff')
    if best_ratio_ft != best_ratio_no_ft:
        plt.axvline(x=best_ratio_ft*100, color='g', linestyle='-.', alpha=0.5, label='Best FT Tradeoff')
    
    plt.title('Impact of Structural Pruning Ratio on Validation Accuracy')
    plt.xlabel('Channel Pruning Ratio (%)')
    plt.ylabel('Validation Accuracy')
    plt.xticks([r*100 for r in prune_ratios])
    plt.legend()
    plt.grid(True)
    
    plot_path = "struct_pruning_tradeoff_graph.png"
    plt.savefig(plot_path)
    print(f"Saved tradeoff graph to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()