import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# 1. Model Definition
class AttentiveMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 architecture tailored for binary classification 
    (attentive vs. not_attentive).
    """
    def __init__(self):
        super().__init__()
        # Grab the standard MobileNetV2 pre-trained on ImageNet
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        
        # We need to know the input size for our custom head before we strip out the old one
        in_features = backbone.classifier[1].in_features
        
        # Replace the default ImageNet classifier with an empty Identity layer
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        
        # Build our custom classification head with some dropout to prevent overfitting
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

# 2. Helper Functions

def apply_pruning(model, amount=0.3):
    """
    Applies L1 unstructured pruning to all convolutional and linear layers in the model.
    It removes the weights with the smallest absolute values (magnitudes).
    
    Args:
        model (nn.Module): The PyTorch model to prune.
        amount (float): The fraction of weights to zero out (e.g., 0.3 means 30%).
        
    Returns:
        nn.Module: The model with pruning permanently applied.
    """
    for module in model.modules():
        # We only want to prune layers that actually have substantial weights
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model


class ImagePathDataset(Dataset):
    """
    Custom Dataset that loads images from file paths and assigns labels based on the folder/file name.
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
    Reads the data split JSON and creates PyTorch DataLoaders for training and validation.
    
    Args:
        splits_path (str): Path to the JSON file containing train/val file paths.
        img_size (int): The height and width to resize images to.
        batch_size (int): Number of images per batch.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    try:
        with open(splits_path, "r") as f:
            splits = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {splits_path}. Please check the path.")
        return None, None
    
    # Safely get the file lists. Fallback to an empty list if the key doesn't exist.
    train_files = splits.get("train_files", [])
    val_files = splits.get("val_files", splits.get("test_files", [])) 

    # Training gets data augmentation to help the model generalize
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation only gets resized and normalized (no random flipping here!)
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


def evaluate_model(model, dataloader, device):
    """
    Runs a full pass over the provided dataloader to calculate classification accuracy.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The validation/test dataloader.
        device (torch.device): CPU or GPU.
        
    Returns:
        float: Accuracy as a decimal (e.g., 0.85 for 85%).
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


def finetune_model(model, train_loader, device, epochs=3, lr=1e-4):
    """
    Trains the model for a few epochs. Used to help the model "recover" accuracy 
    after pruning has destroyed some of its weights.
    
    Args:
        model (nn.Module): The pruned model to fine-tune.
        train_loader (DataLoader): Training data.
        device (torch.device): CPU or GPU.
        epochs (int): Number of passes over the training data.
        lr (float): Learning rate.
        
    Returns:
        nn.Module: The fine-tuned model.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
    return model

# 3. Main Search & Plotting Logic

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    if train_loader is None or len(val_loader.dataset) == 0:
        print("Warning: Missing training or validation data. Check dataset_splits.json.")
        return

    print("\nLoading original baseline model...")
    base_model = AttentiveMobileNetV2()
    try:
        checkpoint = torch.load("attentive_model.pth", map_location=device)
        base_model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    except FileNotFoundError:
        print("Warning: attentive_model.pth not found. Proceeding with uninitialized weights.")
    
    base_model.to(device)

    baseline_acc = evaluate_model(base_model, val_loader, device)
    print(f"Baseline Validation Accuracy: {baseline_acc:.4f}")

    prune_ratios = [i/10.0 for i in range(1, 10)] # Tests [0.1, 0.2, 0.3 ... 0.9]
    acc_no_ft_list = []
    acc_ft_list = []
    
    best_tradeoff_no_ft, best_ratio_no_ft, best_model_state_no_ft = -1.0, 0.0, None
    best_tradeoff_ft, best_ratio_ft, best_model_state_ft = -1.0, 0.0, None

    pruning_weight = 0.5 

    for ratio in prune_ratios:
        print(f"\n--- Testing Pruning Ratio: {ratio*100:.0f}% ---")
        
        model_eval = copy.deepcopy(base_model)
        model_eval = apply_pruning(model_eval, amount=ratio)
        
        acc_no_ft = evaluate_model(model_eval, val_loader, device)
        acc_no_ft_list.append(acc_no_ft)
        print(f"Accuracy (No Fine-tuning): {acc_no_ft:.4f}")

        tradeoff_score_no_ft = acc_no_ft + (ratio * pruning_weight)
        if tradeoff_score_no_ft > best_tradeoff_no_ft:
            best_tradeoff_no_ft = tradeoff_score_no_ft
            best_ratio_no_ft = ratio
            best_model_state_no_ft = copy.deepcopy(model_eval.state_dict())
        
        model_ft = copy.deepcopy(base_model)
        model_ft = apply_pruning(model_ft, amount=ratio)
        
        model_ft = finetune_model(model_ft, train_loader, device, epochs=3, lr=5e-5)
        
        acc_ft = evaluate_model(model_ft, val_loader, device)
        acc_ft_list.append(acc_ft)
        print(f"Accuracy (With Fine-tuning): {acc_ft:.4f}")
        
        tradeoff_score_ft = acc_ft + (ratio * pruning_weight)
        if tradeoff_score_ft > best_tradeoff_ft:
            best_tradeoff_ft = tradeoff_score_ft
            best_ratio_ft = ratio
            best_model_state_ft = copy.deepcopy(model_ft.state_dict())

    print(f"\n=======================================================")
    
    save_path_no_ft = f"best_unstructured_pruned_no_ft_{int(best_ratio_no_ft*100)}.pth"
    torch.save(best_model_state_no_ft, save_path_no_ft)
    print(f"Best Tradeoff (No Fine-Tuning) Found at {best_ratio_no_ft*100:.0f}% Sparsity")
    print(f"Saved optimal No-FT model to: {save_path_no_ft}")
    
    print("-------------------------------------------------------")

    save_path_ft = f"best_unstructured_pruned_ft_{int(best_ratio_ft*100)}.pth"
    torch.save(best_model_state_ft, save_path_ft)
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
    
    plt.title('Impact of Unstructured Pruning Ratio on Validation Accuracy')
    plt.xlabel('Pruning Ratio (%)')
    plt.ylabel('Validation Accuracy')
    plt.xticks([r*100 for r in prune_ratios])
    plt.legend()
    plt.grid(True)
    
    plot_path = "unstructured_pruning_tradeoff_graph.png"
    plt.savefig(plot_path)
    print(f"Saved tradeoff graph to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()