"""
Hybrid Ensemble Model: EfficientNet-B4 + ViT-Base
Run this in Google Colab Account 5
Combines CNN (EfficientNet) and Transformer (ViT) features for better performance
"""

# Setup dataset
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

# Install dependencies
!pip install timm

# Import shared utilities
import sys
sys.path.append('/content')
from shared_utilities import *

# Import model-specific libraries
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np

# ============================================================================
# HYBRID ENSEMBLE MODEL DEFINITION
# ============================================================================

class HybridEnsembleModel(nn.Module):
    """
    Hybrid model combining EfficientNet (CNN) and ViT (Transformer)
    Uses late fusion: concatenates features from both architectures
    """
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        
        # EfficientNet-B4 backbone (CNN features)
        self.efficientnet = timm.create_model('efficientnet_b4', pretrained=True)
        # Get feature dimension (EfficientNet in timm uses num_features)
        eff_features = self.efficientnet.num_features
        # Remove classifier
        if hasattr(self.efficientnet, 'classifier'):
            self.efficientnet.classifier = nn.Identity()
        if hasattr(self.efficientnet, 'head'):
            self.efficientnet.head = nn.Identity()
        
        # ViT-Base backbone (Transformer features)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_features = self.vit.head.in_features
        self.vit.head = nn.Identity()  # Remove classifier
        
        # Feature fusion dimension
        combined_features = eff_features + vit_features  # Will be concatenated in forward
        
        # Attention-based fusion (learns which features to emphasize)
        # This improves over simple concatenation (from research paper techniques)
        self.attention = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Two branches: EfficientNet and ViT
            nn.Softmax(dim=1)
        )
        
        # Feature normalization before fusion (improves stability)
        self.eff_norm = nn.LayerNorm(eff_features)
        self.vit_norm = nn.LayerNorm(vit_features)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract CNN features (EfficientNet)
        if hasattr(self.efficientnet, 'forward_features'):
            eff_features = self.efficientnet.forward_features(x)
        else:
            # Fallback: use forward and extract before classifier
            eff_features = self.efficientnet(x)
        
        # Global average pooling for EfficientNet
        if len(eff_features.shape) == 4:
            # 2D feature map: [B, C, H, W] -> [B, C]
            eff_features = eff_features.mean(dim=[2, 3])
        elif len(eff_features.shape) == 3:
            # 1D sequence: [B, L, C] -> [B, C] (average over length)
            eff_features = eff_features.mean(dim=1)
        # If already [B, C], use as is
        
        # Extract Transformer features (ViT)
        if hasattr(self.vit, 'forward_features'):
            vit_features = self.vit.forward_features(x)
        else:
            vit_features = self.vit(x)
        
        # Take CLS token (first token) for ViT
        if len(vit_features.shape) == 3:
            # [B, num_patches+1, hidden_dim] -> [B, hidden_dim]
            vit_features = vit_features[:, 0]  # CLS token
        elif len(vit_features.shape) == 4:
            # 2D feature map (shouldn't happen for ViT, but handle it)
            vit_features = vit_features.mean(dim=[2, 3])
        # If already [B, C], use as is
        
        # Normalize features before fusion (improves stability)
        eff_features = self.eff_norm(eff_features)
        vit_features = self.vit_norm(vit_features)
        
        # Concatenate features for attention computation
        combined_raw = torch.cat([eff_features, vit_features], dim=1)
        
        # Attention-based weighted fusion (learns which features to emphasize)
        # Compute attention weights: [B, 2] where [w_eff, w_vit]
        attention_weights = self.attention(combined_raw)  # [B, 2]
        
        # Apply attention weights to features (broadcast correctly)
        eff_weighted = eff_features * attention_weights[:, 0:1]  # [B, eff_dim]
        vit_weighted = vit_features * attention_weights[:, 1:2]  # [B, vit_dim]
        
        # Concatenate weighted features
        combined_features = torch.cat([eff_weighted, vit_weighted], dim=1)
        
        # Final classification
        return self.classifier(combined_features)
    
    def freeze_backbones(self):
        """Freeze both backbones, train only classifier"""
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze top layers of both backbones"""
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze EfficientNet top blocks
        # EfficientNet in timm has 'blocks' as ModuleList
        if hasattr(self.efficientnet, 'blocks'):
            blocks = self.efficientnet.blocks
            # Get last N blocks
            if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
                for block in list(blocks)[-num_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            else:
                # Try as list
                try:
                    block_list = list(blocks)
                    for block in block_list[-num_layers:]:
                        for param in block.parameters():
                            param.requires_grad = True
                except:
                    # If can't unfreeze specific blocks, unfreeze all EfficientNet
                    for param in self.efficientnet.parameters():
                        param.requires_grad = True
        
        # Unfreeze ViT top blocks
        if hasattr(self.vit, 'blocks'):
            vit_blocks = self.vit.blocks
            for block in vit_blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Main training function for Hybrid Ensemble Model"""
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
    
    # Initialize hybrid model
    # Increased dropout from 0.5 to 0.6 to prevent overfitting
    model = HybridEnsembleModel(num_classes=NUM_CLASSES, dropout=0.6).to(device)
    model.freeze_backbones()
    
    print("="*60)
    print("HYBRID MODEL ARCHITECTURE")
    print("="*60)
    print("Combining:")
    print("  - EfficientNet-B4 (CNN features)")
    print("  - ViT-Base (Transformer features)")
    print("  - Late fusion: Concatenate features → Classifier")
    
    # Use Focal Loss with label smoothing for better handling of hard examples
    # Better than CrossEntropyLoss for balanced datasets with varying difficulty
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    print("✓ Using Focal Loss with label smoothing (alpha=0.25, gamma=2.0, smoothing=0.1)")
    
    # Phase 1: Train classifier only
    NUM_EPOCHS_PHASE1_HYBRID = 10
    NUM_EPOCHS_PHASE2_HYBRID = 10
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    
    print("\n" + "="*60)
    print("PHASE 1: Training Classifier (Both Backbones Frozen)")
    print(f"Using {NUM_EPOCHS_PHASE1_HYBRID} epochs for hybrid model")
    print("="*60)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    OVERFIT_THRESHOLD = 12.0  # Stricter: Stop if train-val gap > 12% (was 15%)
    
    for epoch in range(NUM_EPOCHS_PHASE1_HYBRID):
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
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE1_HYBRID}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Train-Val Gap: {train_val_gap:.2f}%")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'hybrid_phase1_best.pt')
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
    model.load_state_dict(torch.load('hybrid_phase1_best.pt'))
    model.unfreeze_top_layers(num_layers=2)
    
    # Separate parameter groups
    eff_params = []
    vit_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'efficientnet' in name and param.requires_grad:
            eff_params.append(param)
        elif 'vit' in name and param.requires_grad:
            vit_params.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
    
    optimizer_phase2 = optim.AdamW([
        {'params': classifier_params, 'lr': 1e-4},
        {'params': eff_params, 'lr': 1e-4},  # Increased from 1e-5 to 1e-4
        {'params': vit_params, 'lr': 1e-4}   # Increased from 1e-5 to 1e-4 (better fine-tuning)
    ], weight_decay=1e-4)
    print("✓ Phase 2 LRs: Classifier=1e-4, EfficientNet=1e-4, ViT=1e-4 (increased for better fine-tuning)")
    
    # Improved learning rate scheduling: Cosine Annealing with Warm Restarts
    # Better than ReduceLROnPlateau for fine-tuning (from research paper)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_phase2, T_0=3, T_mult=2, eta_min=1e-6
    )
    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Both Backbones")
    print(f"Using {NUM_EPOCHS_PHASE2_HYBRID} epochs for hybrid model")
    print("="*60)
    
    best_val_loss_phase2 = float('inf')
    patience_counter_phase2 = 0
    OVERFIT_THRESHOLD_PHASE2 = 12.0  # Stricter: Stop if train-val gap > 12% (was 15%)
    
    for epoch in range(NUM_EPOCHS_PHASE2_HYBRID):
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
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE2_HYBRID}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Train-Val Gap: {train_val_gap:.2f}%")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            patience_counter_phase2 = 0
            torch.save(model.state_dict(), 'hybrid_final.pt')
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
    model.load_state_dict(torch.load('hybrid_final.pt'))
    results = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("HYBRID ENSEMBLE (EfficientNet+ViT) FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"MCC: {results['mcc']:.4f}")
    
    # Visualizations
    plot_confusion_matrix(
        results['labels'], results['predictions'], 
        'Hybrid (EfficientNet+ViT)', save_path='hybrid_cm.png'
    )
    
    # Extract features for t-SNE (from combined features)
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
                
                # Extract combined features
                eff_feat = model.efficientnet.forward_features(images)
                vit_feat = model.vit.forward_features(images)
                
                # Process features
                if len(eff_feat.shape) == 4:
                    eff_feat = eff_feat.mean(dim=[2, 3])
                elif len(eff_feat.shape) == 3:
                    eff_feat = eff_feat.mean(dim=1)
                
                if len(vit_feat.shape) == 3:
                    vit_feat = vit_feat[:, 0]
                
                # Concatenate
                combined = torch.cat([eff_feat, vit_feat], dim=1)
                features_list.append(combined.cpu().numpy())
                labels_list.extend(labels.numpy())
                count += len(labels)
        
        if features_list:
            features_array = np.vstack(features_list)
            labels_array = np.array(labels_list[:len(features_array)])
            plot_tsne(features_array, labels_array, 'Hybrid (EfficientNet+ViT)', 
                     save_path='hybrid_tsne.png')
    except Exception as e:
        print(f"Could not extract features for t-SNE: {e}")
    
    # Save model and results (local + Google Drive)
    # save_model_and_results is already imported via 'from shared_utilities import *'
    save_model_and_results(model, results, 'hybrid', save_to_drive=True)
    
    print("\n✓ Hybrid model training completed!")
    print("✓ Model and results saved locally and to Google Drive")
    
    return model, results, history

if __name__ == '__main__':
    model, results, history = train_model()

