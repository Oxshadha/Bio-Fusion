"""
Vision Transformer (ViT) Model Training for Google Colab
With Improvements: Stain Normalization, Focal Loss, Class Weighting
20 Epochs (10 Phase 1 + 10 Phase 2)

Copy each cell into a separate cell in your Colab notebook.
"""

# ============================================================================
# CELL 1: Setup Kaggle and Download Dataset
# ============================================================================

# Upload kaggle.json file
from google.colab import files
files.upload()  # Upload kaggle.json

# Setup Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

print("✓ Dataset downloaded and extracted")

# ============================================================================
# CELL 2: Install Required Packages
# ============================================================================

!pip install timm transformers torchstain scikit-learn matplotlib seaborn

print("✓ All packages installed")

# ============================================================================
# CELL 3: Mount Google Drive (Optional - for saving models)
# ============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("✓ Google Drive mounted")

# Create model directory
import os
model_dir = '/content/drive/MyDrive/BioFusion_Models'
os.makedirs(model_dir, exist_ok=True)
print(f"✓ Model directory ready: {model_dir}")

# ============================================================================
# CELL 4: Import Libraries
# ============================================================================

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
import timm

# Add shared utilities path
sys.path.append('/content')
# Upload shared_utilities.py to Colab first, or copy its contents
from shared_utilities import (
    set_seeds, get_device, load_dataset_paths, create_splits,
    HistopathologyDataset, train_one_epoch, validate, evaluate_model,
    plot_confusion_matrix, plot_tsne, CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)

# Colab dataset path
DATASET_PATH = "/content/GCHTID/HMU-GC-HE-30K/all_image"

print(f"✓ Dataset path: {DATASET_PATH}")
print(f"✓ Classes: {CLASSES}")
print(f"✓ Number of classes: {NUM_CLASSES}")

# ============================================================================
# CELL 5: Improvements Implementation
# ============================================================================

# STAIN NORMALIZATION
def get_stain_normalized_transform():
    """Get transform with stain normalization using Macenko method"""
    try:
        from torchstain import MacenkoNormalizer
        
        normalizer = MacenkoNormalizer(backend='torch')
        
        def normalize_stain(image):
            """Normalize H&E stain"""
            if isinstance(image, Image.Image):
                image_tensor = transforms.ToTensor()(image)
            else:
                image_tensor = image
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            normalized = normalizer.normalize(image_tensor)
            
            if isinstance(image, Image.Image):
                normalized = normalized.squeeze(0)
                normalized = transforms.ToPILImage()(normalized)
            
            return normalized
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.Lambda(normalize_stain),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Lambda(normalize_stain),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✓ Stain normalization enabled (Macenko method)")
        return train_transform, val_test_transform
    
    except ImportError:
        print("⚠ Warning: torchstain not installed. Using standard transforms.")
        from shared_utilities import get_transforms
        return get_transforms(augment=True), get_transforms(augment=False)

# FOCAL LOSS
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# CLASS-SPECIFIC LOSS WEIGHTING
def get_class_weights_for_confusion(y_train, confusion_pairs=None):
    """Create class weights that penalize specific confusions"""
    base_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    if confusion_pairs:
        for class1_idx, class2_idx in confusion_pairs:
            base_weights[class1_idx] *= 1.5
            base_weights[class2_idx] *= 1.5
    
    return torch.FloatTensor(base_weights)

class WeightedFocalLoss(nn.Module):
    """Focal Loss with class weights"""
    def __init__(self, class_weights, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

print("✓ Improvements loaded: Stain Normalization, Focal Loss, Class Weighting")

# ============================================================================
# CELL 6: ViT Model Definition
# ============================================================================

class UNIClassifier(nn.Module):
    """Vision Transformer (ViT) Classifier"""
    def __init__(self, num_classes=8, dropout=0.5, use_uni=False):
        super().__init__()
        
        self.use_hf_model = False
        
        if use_uni:
            try:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained('microsoft/uni')
                if hasattr(self.backbone, 'config'):
                    hidden_size = self.backbone.config.hidden_size
                    self.use_hf_model = True
                    print("✓ Loaded UNI from Hugging Face")
                else:
                    raise ValueError("UNI model structure not recognized")
            except Exception as e:
                print(f"Could not load UNI from Hugging Face: {e}")
                print("Falling back to ViT-Base")
                use_uni = False
        
        if not self.use_hf_model:
            try:
                self.backbone = timm.create_model('uni_vit_base', pretrained=True)
                hidden_size = self.backbone.num_features
                print("✓ Loaded UNI from timm")
            except:
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                hidden_size = self.backbone.num_features
                print("✓ Loaded ViT-Base from timm")
        
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, x):
        """Forward pass for ViT model"""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        if isinstance(features, tuple):
            features = features[-1]
        
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        elif len(features.shape) == 4:
            features = features.mean(dim=[2, 3])
        
        return self.classifier(features)
    
    def freeze_backbone(self):
        """Freeze backbone, train only classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze top transformer layers for fine-tuning"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, 'encoder'):
            encoder_layers = self.backbone.encoder.layer
            for layer in encoder_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

print("✓ ViT Model class defined")

# ============================================================================
# CELL 7: Configuration and Setup
# ============================================================================

# Training hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS_PHASE1 = 10  # Phase 1: Train classifier head
NUM_EPOCHS_PHASE2 = 10  # Phase 2: Fine-tune transformer layers
EARLY_STOP_PATIENCE = 5

# Set seeds for reproducibility
set_seeds(RANDOM_SEED)

# Get device
device = get_device()

# GPU Verification
print("\n" + "="*60)
print("GPU VERIFICATION")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"✓ Device: {device}")
    print("\n🚀 Training will use GPU!")

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"✓ Random seed: {RANDOM_SEED}")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Phase 1 epochs: {NUM_EPOCHS_PHASE1}")
print(f"✓ Phase 2 epochs: {NUM_EPOCHS_PHASE2}")
print(f"✓ Early stopping patience: {EARLY_STOP_PATIENCE}")

# ============================================================================
# CELL 8: Load Dataset and Create Splits
# ============================================================================

# Load dataset paths
image_paths, labels = load_dataset_paths(dataset_root=DATASET_PATH)

# Create train/val/test splits
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(
    image_paths, labels, test_size=0.15, val_size=0.15, seed=RANDOM_SEED
)

print(f"\n✓ Dataset loaded and split successfully")
print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# ============================================================================
# CELL 9: Setup Data Loaders with Stain Normalization
# ============================================================================

# Get transforms with stain normalization
train_transform, val_test_transform = get_stain_normalized_transform()

# Create datasets
train_dataset = HistopathologyDataset(X_train, y_train, transform=train_transform)
val_dataset = HistopathologyDataset(X_val, y_val, transform=val_test_transform)
test_dataset = HistopathologyDataset(X_test, y_test, transform=val_test_transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, 
    shuffle=False, num_workers=2
)

print(f"✓ Data loaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# CELL 10: Setup Loss Function with Class Weighting
# ============================================================================

# Define critical confusion pairs to penalize more
confusion_pairs = [
    (CLASS_TO_IDX['NOR'], CLASS_TO_IDX['TUM']),  # Normal vs Tumor
    (CLASS_TO_IDX['MUC'], CLASS_TO_IDX['ADI']),  # Mucus vs Adipose
    (CLASS_TO_IDX['DEB'], CLASS_TO_IDX['STR']),  # Debris vs Stroma
]

# Get class weights
class_weights = get_class_weights_for_confusion(y_train, confusion_pairs)

# Create Weighted Focal Loss
criterion = WeightedFocalLoss(class_weights, alpha=0.25, gamma=2.0).to(device)

print("✓ Loss function: Weighted Focal Loss")
print(f"  Class weights: {dict(zip(CLASSES, class_weights.cpu().numpy()))}")
print(f"  Confusion pairs penalized: {[(CLASSES[i], CLASSES[j]) for i, j in confusion_pairs]}")

# ============================================================================
# CELL 11: Initialize Model
# ============================================================================

# Initialize ViT model
model = UNIClassifier(num_classes=NUM_CLASSES, dropout=0.5, use_uni=False).to(device)
print("✓ Model initialized and moved to device")

# Freeze backbone for Phase 1
model.freeze_backbone()
print("✓ Backbone frozen (Phase 1: training classifier only)")

# ============================================================================
# CELL 12: Phase 1 - Train Classifier Head
# ============================================================================

# Phase 1: Train classifier head
optimizer_phase1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)

print("="*60)
print("PHASE 1: Training Classifier Head (ViT Backbone Frozen)")
print("="*60)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS_PHASE1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer_phase1, device
    )
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE1}]")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'vit_phase1_best.pt')
        print(f"  ✓ Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping")
            break

print("\n✓ Phase 1 completed!")

# ============================================================================
# CELL 13: Phase 2 - Fine-tune Transformer Layers
# ============================================================================

# Load best model from Phase 1
model.load_state_dict(torch.load('vit_phase1_best.pt'))
model.unfreeze_top_layers(num_layers=2)

# Separate parameter groups for different learning rates
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'classifier' in name or 'head' in name:
        head_params.append(param)
    elif param.requires_grad:
        backbone_params.append(param)

optimizer_phase2 = optim.AdamW([
    {'params': head_params, 'lr': 1e-4},
    {'params': backbone_params, 'lr': 1e-5}  # Lower LR for pretrained transformer
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_phase2, mode='min', factor=0.5, patience=3
)

print("\n" + "="*60)
print("PHASE 2: Fine-tuning ViT Transformer Layers")
print("="*60)

best_val_loss_phase2 = float('inf')
patience_counter_phase2 = 0

for epoch in range(NUM_EPOCHS_PHASE2):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer_phase2, device
    )
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE2}]")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    print(f"  LR: {optimizer_phase2.param_groups[0]['lr']:.2e}")
    
    if val_loss < best_val_loss_phase2:
        best_val_loss_phase2 = val_loss
        patience_counter_phase2 = 0
        torch.save(model.state_dict(), 'vit_final.pt')
        print(f"  ✓ Best model saved")
    else:
        patience_counter_phase2 += 1
        if patience_counter_phase2 >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping")
            break

print("\n✓ Phase 2 completed!")

# ============================================================================
# CELL 14: Final Evaluation on Test Set
# ============================================================================

# Load best model
model.load_state_dict(torch.load('vit_final.pt'))

# Evaluate on test set
results = evaluate_model(model, test_loader, criterion, device)

print("\n" + "="*60)
print("VIT-BASE FINAL RESULTS")
print("="*60)
print(f"Test Accuracy: {results['test_acc']:.2f}%")
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"Weighted F1: {results['weighted_f1']:.4f}")
print(f"MCC: {results['mcc']:.4f}")

print("\nPer-Class F1 Scores:")
for i, class_name in enumerate(CLASSES):
    print(f"  {class_name}: {results['per_class_f1'][i]:.4f}")

# Print detailed classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    results['labels'], results['predictions'],
    target_names=CLASSES,
    digits=4
))

# ============================================================================
# CELL 15: Visualizations
# ============================================================================

# Plot confusion matrix
plot_confusion_matrix(
    results['labels'], results['predictions'], 
    'ViT-Base', save_path='vit_cm.png'
)

# Extract and plot t-SNE (ViT embeddings)
try:
    model.eval()
    features_list = []
    labels_list = []
    count = 0
    max_samples = 500
    
    with torch.no_grad():
        for images, labels in test_loader:
            if count >= max_samples:
                break
            images = images.to(device)
            
            # Extract features before classifier
            if hasattr(model.backbone, 'forward_features'):
                feat = model.backbone.forward_features(images)
            else:
                feat = model.backbone(images)
            
            # Handle tuple output
            if isinstance(feat, tuple):
                feat = feat[-1]
            
            # Take CLS token (first token) for ViT
            if len(feat.shape) == 3:
                feat = feat[:, 0]  # [batch, hidden_dim]
            
            features_list.append(feat.cpu().numpy())
            labels_list.extend(labels.numpy())
            count += len(labels)
    
    if features_list:
        features_array = np.vstack(features_list)
        labels_array = np.array(labels_list[:len(features_array)])
        plot_tsne(features_array, labels_array, 'ViT-Base', save_path='vit_tsne.png')
        print("✓ t-SNE visualization saved")
except Exception as e:
    print(f"⚠ Could not extract features for t-SNE: {e}")
    print("  This is optional - training and evaluation completed successfully")

# ============================================================================
# CELL 16: Save Model and Results
# ============================================================================

# Save model weights locally
model_path = 'vit_final.pt'
torch.save(model.state_dict(), model_path)
print(f"✓ Model saved: {model_path}")

# Save results locally
results_path = 'vit_results.json'
with open(results_path, 'w') as f:
    json.dump({
        'test_acc': float(results['test_acc']),
        'macro_f1': float(results['macro_f1']),
        'weighted_f1': float(results['weighted_f1']),
        'mcc': float(results['mcc']),
        'per_class_f1': results['per_class_f1'].tolist(),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        }
    }, f, indent=2)
print(f"✓ Results saved: {results_path}")

# Save to Google Drive
try:
    drive_model_path = os.path.join(model_dir, 'vit_final.pt')
    drive_results_path = os.path.join(model_dir, 'vit_results.json')
    
    torch.save(model.state_dict(), drive_model_path)
    with open(drive_results_path, 'w') as f:
        json.dump({
            'test_acc': float(results['test_acc']),
            'macro_f1': float(results['macro_f1']),
            'weighted_f1': float(results['weighted_f1']),
            'mcc': float(results['mcc']),
            'per_class_f1': results['per_class_f1'].tolist(),
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']]
            }
        }, f, indent=2)
    
    print(f"✓ Model saved to Google Drive: {drive_model_path}")
    print(f"✓ Results saved to Google Drive: {drive_results_path}")
except Exception as e:
    print(f"⚠ Could not save to Google Drive: {e}")

print("\n" + "="*60)
print("✓ ViT-Base training completed successfully!")
print("="*60)
print(f"✓ Model: {model_path}")
print(f"✓ Results: {results_path}")
print(f"✓ Confusion Matrix: vit_cm.png")
print(f"✓ t-SNE: vit_tsne.png")

# ============================================================================
# CELL 17: Training History Plot (Optional)
# ============================================================================

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=NUM_EPOCHS_PHASE1-1, color='r', linestyle='--', alpha=0.5, label='Phase 1/2 Split')
axes[0].legend()

# Accuracy plot
axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=NUM_EPOCHS_PHASE1-1, color='r', linestyle='--', alpha=0.5, label='Phase 1/2 Split')
axes[1].legend()

plt.tight_layout()
plt.savefig('vit_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training history plot saved: vit_training_history.png")

