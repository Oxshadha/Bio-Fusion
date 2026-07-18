"""
ResNet50 Baseline Model Training
Run this in Google Colab Account 1
"""

# Setup dataset (run once)
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

# Import shared utilities
import sys
sys.path.append('/content')
from shared_utilities import *

# Import model-specific libraries
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        layers_to_unfreeze = list(self.backbone.children())[-num_layers:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """Main training function"""
    # Setup
    set_seeds()
    device = get_device()
    
    # Load data
    image_paths, labels = load_dataset_paths()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(
        image_paths, labels
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Model
    model = ResNet50Classifier(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    model.freeze_backbone()
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Phase 1: Train head only
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    
    print("="*60)
    print("PHASE 1: Training Classifier Head")
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
            torch.save(model.state_dict(), 'resnet50_phase1_best.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping")
                break
    
    # Phase 2: Fine-tuning
    model.load_state_dict(torch.load('resnet50_phase1_best.pt'))
    model.unfreeze_top_layers(num_layers=2)
    
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            head_params.append(param)
        elif param.requires_grad:
            backbone_params.append(param)
    
    optimizer_phase2 = optim.AdamW([
        {'params': head_params, 'lr': 1e-4},
        {'params': backbone_params, 'lr': 1e-4}  # Increased from 1e-5
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2, mode='min', factor=0.5, patience=3
    )
    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning")
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
            torch.save(model.state_dict(), 'resnet50_final.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter_phase2 += 1
            if patience_counter_phase2 >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load('resnet50_final.pt'))
    results = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("RESNET50 FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"MCC: {results['mcc']:.4f}")
    
    # Visualizations
    plot_confusion_matrix(
        results['labels'], results['predictions'], 
        'ResNet50', save_path='resnet50_cm.png'
    )
    
    # Save model and results (local + Google Drive)
    # save_model_and_results is already imported via 'from shared_utilities import *'
    save_model_and_results(model, results, 'resnet50', save_to_drive=True)
    
    return model, results, history

if __name__ == '__main__':
    model, results, history = train_model()

