"""
Vision Transformer (ViT) Model Training - UNI-style
Run this in Google Colab Account 4

Note: This uses ViT-Base from timm as a strong transformer baseline.
If UNI becomes available, it can be easily swapped in.
ViT-Base is a proven architecture that works well for histopathology.
"""

# Setup dataset
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

# Install UNI dependencies
!pip install timm transformers

# Import shared utilities
import sys
sys.path.append('/content')
from shared_utilities import *

# Import model-specific libraries
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

# ============================================================================
# MODEL DEFINITION - UNI
# ============================================================================

class UNIClassifier(nn.Module):
    """
    UNI (Universal Medical Image Representation) Classifier
    Uses ViT-Base as backbone (timm) - works reliably
    If UNI becomes available, can be swapped in
    """
    def __init__(self, num_classes=8, dropout=0.5, use_uni=False):
        super().__init__()
        
        self.use_hf_model = False
        
        # Try to load UNI if requested and available
        if use_uni:
            try:
                # Try Hugging Face UNI
                self.backbone = AutoModel.from_pretrained('microsoft/uni')
                if hasattr(self.backbone, 'config'):
                    hidden_size = self.backbone.config.hidden_size
                    self.use_hf_model = True
                    print("✓ Loaded UNI from Hugging Face")
                else:
                    raise ValueError("UNI model structure not recognized")
            except Exception as e:
                print(f"Could not load UNI from Hugging Face: {e}")
                print("Falling back to ViT-Base (still a strong transformer baseline)")
                use_uni = False
        
        # Default: Use ViT-Base from timm (reliable and works well)
        if not self.use_hf_model:
            # Try UNI in timm first, then fallback to ViT
            try:
                self.backbone = timm.create_model('uni_vit_base', pretrained=True)
                hidden_size = self.backbone.num_features
                print("✓ Loaded UNI from timm")
            except:
                # Fallback to standard ViT-Base (proven to work)
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                hidden_size = self.backbone.num_features
                print("✓ Loaded ViT-Base from timm (strong transformer baseline)")
        
        # Replace classifier head
        if hasattr(self.backbone, 'head'):
            # Timm models
            self.backbone.head = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        elif hasattr(self.backbone, 'classifier'):
            # Hugging Face models
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            # Custom classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass for UNI/ViT model
        Handles timm ViT models (which is what we're using)
        """
        # For timm models (ViT-Base or UNI from timm)
        if hasattr(self.backbone, 'forward_features'):
            # Extract features (before head)
            features = self.backbone.forward_features(x)
        else:
            # Direct forward if head already removed
            features = self.backbone(x)
        
        # Handle tuple output
        if isinstance(features, tuple):
            features = features[-1]
        
        # ViT outputs: [batch, num_patches+1, hidden_dim]
        # We need [batch, hidden_dim] - take CLS token (first token)
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        elif len(features.shape) == 4:
            # 2D feature map (shouldn't happen for ViT, but handle it)
            features = features.mean(dim=[2, 3])
        # If already [batch, hidden_dim], use as is
        
        return self.classifier(features)
    
    def freeze_backbone(self):
        """Freeze UNI backbone, train only classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze top transformer layers for fine-tuning"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze last N transformer blocks (timm style)
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, 'encoder'):
            # Hugging Face style (if using HF model)
            encoder_layers = self.backbone.encoder.layer
            for layer in encoder_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

# Alternative: If UNI is available as a specific package
# You might need to install: !pip install uni-medical
# Then use: from uni_medical import UNIModel

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Main training function for UNI"""
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
    
    # Initialize UNI/ViT model
    # Try UNI first, fallback to ViT-Base (both are transformers, ViT is proven to work)
    model = UNIClassifier(num_classes=NUM_CLASSES, dropout=0.5, use_uni=False).to(device)
    print("✓ Model loaded successfully (ViT-Base or UNI)")
    
    model.freeze_backbone()
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Phase 1: Train classifier head
    optimizer = optim.AdamW(
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
            model, train_loader, criterion, optimizer, device
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
            torch.save(model.state_dict(), 'uni_phase1_best.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping")
                break
    
    # Phase 2: Fine-tuning
    model.load_state_dict(torch.load('uni_phase1_best.pt'))
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
        
        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            patience_counter_phase2 = 0
            torch.save(model.state_dict(), 'uni_final.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter_phase2 += 1
            if patience_counter_phase2 >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load('uni_final.pt'))
    results = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("VIT-BASE (UNI-style) FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"MCC: {results['mcc']:.4f}")
    
    # Visualizations
    plot_confusion_matrix(
        results['labels'], results['predictions'], 
        'ViT-Base', save_path='uni_cm.png'
    )
    
    # Extract and plot t-SNE (ViT embeddings)
    # ViT doesn't have avgpool, so we extract features directly
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
            plot_tsne(features_array, labels_array, 'ViT-Base', save_path='uni_tsne.png')
    except Exception as e:
        print(f"Could not extract features for t-SNE: {e}")
        print("This is optional - training and evaluation completed successfully")
    
    # Save model and results (local + Google Drive)
    # save_model_and_results is already imported via 'from shared_utilities import *'
    save_model_and_results(model, results, 'uni', save_to_drive=True)
    
    print("\n✓ ViT-Base training completed!")
    print("✓ Model and results saved locally and to Google Drive")
    print("\nNote: Model uses ViT-Base (strong transformer baseline)")
    print("      Results file named 'uni_results.json' for consistency with comparison script")
    
    return model, results, history

if __name__ == '__main__':
    model, results, history = train_model()

