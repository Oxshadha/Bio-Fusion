"""
Shared utilities for multi-model training in Google Colab
Avoids code duplication across different model experiments
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, matthews_corrcoef
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch.nn.functional as F
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path for Google Colab
DATASET_PATH = "/content/GCHTID/HMU-GC-HE-30K/all_image"

# Model saving directory (Google Drive)
MODEL_SAVE_DIR = "/content/drive/MyDrive/BioFusion_Models"

# Classes
CLASSES = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Training hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS_PHASE1 = 10
NUM_EPOCHS_PHASE2 = 10
EARLY_STOP_PATIENCE = 5

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Get the best available device (CUDA > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU")
    return device

# ============================================================================
# GOOGLE DRIVE SETUP (for model persistence)
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and create model directory"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive mounted")
    except Exception as e:
        print(f"⚠ Google Drive mount issue: {e}")
        print("  Models will be saved locally only")
        return None
    
    # Create model directory
    model_dir = '/content/drive/MyDrive/BioFusion_Models'
    os.makedirs(model_dir, exist_ok=True)
    print(f"✓ Model directory ready: {model_dir}")
    return model_dir

def save_model_and_results(model, results, model_name, save_to_drive=True):
    """
    Save model weights and results to both local and Google Drive
    
    Args:
        model: Trained PyTorch model
        results: Results dictionary
        model_name: Name for the model (e.g., 'resnet50', 'vit_base')
        save_to_drive: Whether to save to Google Drive
    """
    # Save model weights locally
    local_model_path = f'{model_name}_final.pt'
    torch.save(model.state_dict(), local_model_path)
    print(f"✓ Model saved locally: {local_model_path}")
    
    # Save results locally
    local_results_path = f'{model_name}_results.json'
    with open(local_results_path, 'w') as f:
        json.dump({
            'test_acc': float(results['test_acc']),
            'macro_f1': float(results['macro_f1']),
            'weighted_f1': float(results['weighted_f1']),
            'mcc': float(results['mcc']),
            'per_class_f1': results['per_class_f1'].tolist()
        }, f, indent=2)
    print(f"✓ Results saved locally: {local_results_path}")
    
    # Save to Google Drive if available
    if save_to_drive:
        try:
            model_dir = setup_google_drive()
            if model_dir:
                drive_model_path = os.path.join(model_dir, f'{model_name}_final.pt')
                drive_results_path = os.path.join(model_dir, f'{model_name}_results.json')
                
                torch.save(model.state_dict(), drive_model_path)
                with open(drive_results_path, 'w') as f:
                    json.dump({
                        'test_acc': float(results['test_acc']),
                        'macro_f1': float(results['macro_f1']),
                        'weighted_f1': float(results['weighted_f1']),
                        'mcc': float(results['mcc']),
                        'per_class_f1': results['per_class_f1'].tolist()
                    }, f, indent=2)
                
                print(f"✓ Model saved to Google Drive: {drive_model_path}")
                print(f"✓ Results saved to Google Drive: {drive_results_path}")
        except Exception as e:
            print(f"⚠ Could not save to Google Drive: {e}")
            print("  Models saved locally only (still accessible)")
    
    return local_model_path, local_results_path

# ============================================================================
# FOCAL LOSS (for hard examples)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for addressing hard examples and class confusion
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Better than CrossEntropyLoss for:
    - Hard examples (NOR, DEB, STR confusion)
    - Balanced datasets with varying difficulty
    
    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0) - higher = more focus on hard examples
        label_smoothing: Label smoothing factor (default: 0.1) - prevents overconfidence
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        # Apply label smoothing (from research paper - prevents overconfidence)
        if self.label_smoothing > 0:
            # Convert targets to one-hot with smoothing
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            
            # Compute log probabilities
            log_probs = F.log_softmax(inputs, dim=1)
            
            # Compute focal loss with label smoothing
            pt = torch.sum(targets_one_hot * torch.exp(log_probs), dim=1)
            ce_loss = -torch.sum(targets_one_hot * log_probs, dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)  # Probability of true class
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_paths(dataset_root=DATASET_PATH):
    """Load all image paths and labels"""
    image_paths = []
    labels = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.exists(class_dir):
            class_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            for filename in class_files:
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(CLASS_TO_IDX[class_name])
    
    print(f"Loaded {len(image_paths)} images from {len(CLASSES)} classes")
    return image_paths, labels

def create_splits(image_paths, labels, test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """Create stratified train/val/test splits"""
    # First split: train vs (val+test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, 
        random_state=seed, stratify=labels
    )
    
    # Second split: val vs test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        random_state=seed, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)} ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"Val: {len(X_val)} ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ============================================================================
# DATASET CLASS
# ============================================================================

class HistopathologyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# TRANSFORMS
# ============================================================================

class CLAHETransform:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) transform
    From research paper: improves medical image quality
    Applied before other transforms
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel separately (for RGB)
        if len(img_np.shape) == 3:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                    tileGridSize=self.tile_grid_size)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            img_np[:, :, 0] = clahe.apply(img_np[:, :, 0])
            img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                    tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        
        # Convert back to PIL
        return Image.fromarray(img_np)

def get_transforms(augment=True, use_clahe=True):
    """
    Get data transforms (with or without augmentation)
    
    Args:
        augment: Whether to apply augmentation (for training)
        use_clahe: Whether to apply CLAHE preprocessing (from research paper)
    """
    transform_list = []
    
    # CLAHE preprocessing (from research paper - improves medical image quality)
    if use_clahe:
        transform_list.append(CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)))
    
    if augment:
        # Training: augmentation
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Reduced from ±90° to prevent overfitting
            # Optional: Add slight color jitter for histopathology stain variation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation/Test: only normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transforms.Compose(transform_list)

# ============================================================================
# DATA LOADERS
# ============================================================================

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=BATCH_SIZE, num_workers=2):
    """Create train/val/test dataloaders"""
    train_transform = get_transforms(augment=True)
    val_test_transform = get_transforms(augment=False)
    
    train_dataset = HistopathologyDataset(X_train, y_train, transform=train_transform)
    val_dataset = HistopathologyDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = HistopathologyDataset(X_test, y_test, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    # Classification report
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        test_labels, test_preds, 
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # Overall metrics
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    weighted_f1 = f1_score(test_labels, test_preds, average='weighted')
    mcc = matthews_corrcoef(test_labels, test_preds)
    
    # Per-class F1
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mcc': mcc,
        'per_class_f1': per_class_f1,
        'predictions': test_preds,
        'labels': test_labels,
        'classification_report': report
    }
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)', 
                     fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_class_f1(per_class_f1_dict, save_path=None):
    """Plot per-class F1 scores for all models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASSES))
    width = 0.2
    offset = 0
    
    for model_name, f1_scores in per_class_f1_dict.items():
        ax.bar(x + offset, f1_scores, width, label=model_name)
        offset += width
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(per_class_f1_dict) - 1) / 2)
    ax.set_xticklabels(CLASSES)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_class_metrics(y_true, y_pred, model_name, save_path=None):
    """Plot comprehensive per-class metrics (Precision, Recall, F1)"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), average=None
    )
    
    x = np.arange(len(CLASSES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Tissue Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed metrics table
    print("\n" + "="*80)
    print(f"PER-CLASS METRICS - {model_name}")
    print("="*80)
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    print("="*80)

def plot_training_curves(history, model_name, phase1_epochs=None, save_path=None):
    """Plot comprehensive training curves (loss and accuracy)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', 
                linewidth=2, color='#3498db', marker='o', markersize=4)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', 
                linewidth=2, color='#e74c3c', marker='s', markersize=4)
    if phase1_epochs:
        axes[0].axvline(x=phase1_epochs, color='green', linestyle='--', 
                       linewidth=2, label='Phase 1 → Phase 2', alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Training and Validation Loss', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], label='Train Accuracy', 
                linewidth=2, color='#3498db', marker='o', markersize=4)
    axes[1].plot(epochs, history['val_acc'], label='Val Accuracy', 
                linewidth=2, color='#e74c3c', marker='s', markersize=4)
    if phase1_epochs:
        axes[1].axvline(x=phase1_epochs, color='green', linestyle='--', 
                       linewidth=2, label='Phase 1 → Phase 2', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Training and Validation Accuracy', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def extract_features(model, test_loader, device, num_samples=500):
    """Extract features from penultimate layer"""
    model.eval()
    features_list = []
    labels_list = []
    
    # Hook to extract features
    def get_features_hook(module, input, output):
        features_list.append(output.view(output.size(0), -1).cpu().numpy())
    
    # Register hook (assumes model has backbone.avgpool)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'avgpool'):
        hook = model.backbone.avgpool.register_forward_hook(get_features_hook)
    elif hasattr(model, 'avgpool'):
        hook = model.avgpool.register_forward_hook(get_features_hook)
    else:
        print("Warning: Could not find avgpool layer for feature extraction")
        return None, None
    
    count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if count >= num_samples:
                break
            images = images.to(device)
            _ = model(images)
            labels_list.extend(labels.numpy())
            count += len(labels)
    
    hook.remove()
    
    if features_list:
        features_array = np.vstack(features_list)
        labels_array = np.array(labels_list[:len(features_array)])
        return features_array, labels_array
    return None, None

def plot_tsne(features, labels, model_name, save_path=None):
    """Plot t-SNE visualization"""
    from sklearn.manifold import TSNE
    
    print(f"Computing t-SNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    for class_idx in range(NUM_CLASSES):
        mask = labels == class_idx
        if np.sum(mask) > 0:
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       label=IDX_TO_CLASS[class_idx], alpha=0.6, s=50)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - t-SNE Visualization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(results_dict, save_path=None):
    """Compare multiple models' performance"""
    models = list(results_dict.keys())
    metrics = ['test_acc', 'macro_f1', 'weighted_f1', 'mcc']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        axes[idx].bar(models, values, color='steelblue', edgecolor='black')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), 
                             fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.4f}', 
                          ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'MCC':<12}")
    print("-"*80)
    for model in models:
        r = results_dict[model]
        print(f"{model:<20} {r['test_acc']:<12.4f} {r['macro_f1']:<12.4f} "
              f"{r['weighted_f1']:<12.4f} {r['mcc']:<12.4f}")
    print("="*80)

