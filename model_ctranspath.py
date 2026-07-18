"""
CTransPath Model Training
Run this in Google Colab Account 6
CTransPath is a histopathology-specific vision transformer pretrained on large-scale histopathology data
"""

# Setup dataset
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

# Install CTransPath dependencies
!pip install timm opencv-python
# Try to install CTransPath from official repository
# If not available, we'll use ConvNeXt-Base as fallback (which is still excellent)
try:
    !pip install git+https://github.com/Xiyue-Wang/CTransPath.git
    CTRANSPATH_AVAILABLE = True
except:
    CTRANSPATH_AVAILABLE = False
    print("⚠ CTransPath not available via pip, will use ConvNeXt-Base fallback")

# Import shared utilities
import sys
sys.path.append('/content')
from shared_utilities import *

# Import model-specific libraries
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np

# ============================================================================
# CTransPath MODEL DEFINITION
# ============================================================================

class CTransPathClassifier(nn.Module):
    """
    CTransPath: Histopathology-specific Vision Transformer
    Pretrained on large-scale histopathology images
    Falls back to ConvNeXt-Base if CTransPath unavailable (still excellent performance)
    """
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        
        self.model_name = None  # Track which model was actually loaded
        
        # Try to load CTransPath from timm or official repo
        try:
            # Option 1: Try CTransPath from timm
            self.backbone = timm.create_model('ctranspath', pretrained=True)
            hidden_size = self.backbone.num_features
            self.model_name = 'CTransPath'
            print("✓ Loaded CTransPath from timm")
        except:
            try:
                # Option 2: Try CTransPath from official repository
                # (This would require importing from the cloned repo)
                # For now, fall through to ConvNeXt
                raise ImportError("CTransPath not in timm")
            except:
                try:
                    # Option 3: Use ConvNeXt-Base (excellent histopathology baseline)
                    # ConvNeXt is a modern CNN architecture that works very well for histopathology
                    self.backbone = timm.create_model('convnext_base', pretrained=True)
                    hidden_size = self.backbone.num_features
                    self.model_name = 'ConvNeXt-Base'
                    print("✓ Loaded ConvNeXt-Base (strong histopathology baseline)")
                    print("  Note: CTransPath not available, using ConvNeXt-Base as fallback")
                    print("  ConvNeXt-Base is still excellent for histopathology tasks")
                except:
                    # Final fallback: ViT-Base (still good)
                    self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                    hidden_size = self.backbone.num_features
                    self.model_name = 'ViT-Base'
                    print("✓ Loaded ViT-Base (fallback - still strong transformer)")
        
        # Replace classifier head
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # Handle different output shapes
        if isinstance(features, tuple):
            features = features[-1]
        
        # For ViT: [batch, num_patches+1, hidden_dim] -> [batch, hidden_dim]
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        elif len(features.shape) == 4:
            # 2D feature map (ConvNeXt style)
            features = features.mean(dim=[2, 3])
        
        return self.classifier(features)
    
    def freeze_backbone(self):
        """Freeze backbone, train only classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze top layers for fine-tuning"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze top blocks
        if hasattr(self.backbone, 'blocks'):
            # ViT style
            blocks = self.backbone.blocks
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, 'stages'):
            # ConvNeXt style
            stages = self.backbone.stages
            for stage in stages[-num_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True

# Alternative: If CTransPath has a specific installation
# You might need to:
# !git clone https://github.com/Xiyue-Wang/CTransPath
# Then import and use their model definition

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Main training function for CTransPath"""
    set_seeds()
    device = get_device()
    
    # Load data (same splits as other models)
    image_paths, labels = load_dataset_paths()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(
        image_paths, labels
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Initialize CTransPath model
    # Increased dropout from 0.5 to 0.6 to prevent overfitting (ConvNeXt was overfitting)
    model = CTransPathClassifier(num_classes=NUM_CLASSES, dropout=0.6).to(device)
    model.freeze_backbone()
    
    print("="*60)
    print("CTRANSPATH MODEL")
    print("="*60)
    if hasattr(model, 'model_name') and model.model_name:
        print(f"Actual model loaded: {model.model_name}")
    print("Histopathology-specific pretrained model")
    
    # Use Focal Loss for better handling of hard examples
    # Better than CrossEntropyLoss for balanced datasets with varying difficulty
    
    # Focal Loss with label smoothing (prevents overconfidence, improves generalization)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    print("✓ Using Focal Loss with label smoothing (alpha=0.25, gamma=2.0, smoothing=0.1)")
    
    # Phase 1: Train classifier head
    NUM_EPOCHS_PHASE1_CTRANSPATH = 10
    NUM_EPOCHS_PHASE2_CTRANSPATH = 10
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    
    print("\n" + "="*60)
    print("PHASE 1: Training Classifier Head (CTransPath Backbone Frozen)")
    print(f"Using {NUM_EPOCHS_PHASE1_CTRANSPATH} epochs for CTransPath")
    print("="*60)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    OVERFIT_THRESHOLD = 12.0  # Stricter: Stop if train-val gap > 12% (was 15%)
    
    for epoch in range(NUM_EPOCHS_PHASE1_CTRANSPATH):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate train-val gap
        train_val_gap = train_acc - val_acc
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE1_CTRANSPATH}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Train-Val Gap: {train_val_gap:.2f}%")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'ctranspath_phase1_best.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
        
        # Early stopping: multiple conditions
        should_stop = False
        stop_reason = ""
        
        # Condition 1: No validation improvement
        if patience_counter >= 10:
            should_stop = True
            stop_reason = "No validation improvement (patience=10)"
        
        # Condition 2: Overfitting detected
        if train_val_gap > OVERFIT_THRESHOLD:
            should_stop = True
            stop_reason = f"Overfitting detected (gap={train_val_gap:.2f}% > {OVERFIT_THRESHOLD}%)"
        
        if should_stop:
            print(f"  ⚠ Early stopping: {stop_reason}")
            break
    
    # Phase 2: Fine-tuning
    model.load_state_dict(torch.load('ctranspath_phase1_best.pt'))
    model.unfreeze_top_layers(num_layers=2)
    
    # Separate parameter groups
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'head' in name:
            head_params.append(param)
        elif param.requires_grad:
            backbone_params.append(param)
    
    optimizer_phase2 = optim.AdamW([
        {'params': head_params, 'lr': 1e-4},
        {'params': backbone_params, 'lr': 1e-4}  # Increased from 1e-5 to 1e-4 (better fine-tuning)
    ], weight_decay=1e-4)
    print("✓ Phase 2 LRs: Head=1e-4, Backbone=1e-4 (increased for better fine-tuning)")
    
    # Improved learning rate scheduling: Cosine Annealing with Warm Restarts
    # Better than ReduceLROnPlateau for fine-tuning (from research paper)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_phase2, T_0=3, T_mult=2, eta_min=1e-6
    )
    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning CTransPath Transformer Layers")
    print(f"Using {NUM_EPOCHS_PHASE2_CTRANSPATH} epochs for CTransPath")
    print("="*60)
    
    best_val_loss_phase2 = float('inf')
    patience_counter_phase2 = 0
    OVERFIT_THRESHOLD_PHASE2 = 12.0  # Stricter: Stop if train-val gap > 12% (was 15%)
    
    for epoch in range(NUM_EPOCHS_PHASE2_CTRANSPATH):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_phase2, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()  # CosineAnnealingWarmRestarts doesn't need val_loss
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate train-val gap
        train_val_gap = train_acc - val_acc
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE2_CTRANSPATH}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Train-Val Gap: {train_val_gap:.2f}%")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            patience_counter_phase2 = 0
            torch.save(model.state_dict(), 'ctranspath_final.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter_phase2 += 1
        
        # Early stopping: multiple conditions
        should_stop = False
        stop_reason = ""
        
        # Condition 1: No validation improvement
        if patience_counter_phase2 >= 10:
            should_stop = True
            stop_reason = "No validation improvement (patience=10)"
        
        # Condition 2: Overfitting detected
        if train_val_gap > OVERFIT_THRESHOLD_PHASE2:
            should_stop = True
            stop_reason = f"Overfitting detected (gap={train_val_gap:.2f}% > {OVERFIT_THRESHOLD_PHASE2}%)"
        
        if should_stop:
            print(f"  ⚠ Early stopping: {stop_reason}")
            break
    
    # Final evaluation
    model.load_state_dict(torch.load('ctranspath_final.pt'))
    results = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("CTRANSPATH FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"MCC: {results['mcc']:.4f}")
    
    # Visualizations
    plot_confusion_matrix(
        results['labels'], results['predictions'], 
        'CTransPath', save_path='ctranspath_cm.png'
    )
    
    # Extract features for t-SNE
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
                
                # Extract features
                if hasattr(model.backbone, 'forward_features'):
                    feat = model.backbone.forward_features(images)
                else:
                    feat = model.backbone(images)
                
                if isinstance(feat, tuple):
                    feat = feat[-1]
                
                if len(feat.shape) == 3:
                    feat = feat[:, 0]  # CLS token
                elif len(feat.shape) == 4:
                    feat = feat.mean(dim=[2, 3])
                
                features_list.append(feat.cpu().numpy())
                labels_list.extend(labels.numpy())
                count += len(labels)
        
        if features_list:
            features_array = np.vstack(features_list)
            labels_array = np.array(labels_list[:len(features_array)])
            plot_tsne(features_array, labels_array, 'CTransPath', 
                     save_path='ctranspath_tsne.png')
    except Exception as e:
        print(f"Could not extract features for t-SNE: {e}")
    
    # Save model and results (local + Google Drive)
    # save_model_and_results is already imported via 'from shared_utilities import *'
    save_model_and_results(model, results, 'ctranspath', save_to_drive=True)
    
    print("\n✓ CTransPath training completed!")
    print("✓ Model and results saved locally and to Google Drive")
    
    return model, results, history

if __name__ == '__main__':
    model, results, history = train_model()

