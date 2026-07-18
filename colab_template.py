"""
Quick Start Template for Google Colab
Copy this into a new Colab cell and modify for your model
"""

# ============================================================================
# STEP 1: Setup Dataset (Run Once)
# ============================================================================
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID

# ============================================================================
# STEP 2: Upload Shared Utilities
# ============================================================================
files.upload()  # Upload shared_utilities.py
import sys
sys.path.append('/content')

# ============================================================================
# STEP 3: Import Everything
# ============================================================================
from shared_utilities import *
import torch.nn as nn
import torch.optim as optim
from torchvision import models  # or your model library

# ============================================================================
# STEP 4: Define Your Model (MODIFY THIS SECTION)
# ============================================================================
class YourModelClassifier(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        # TODO: Define your model architecture here
        # Example:
        # self.backbone = models.resnet50(pretrained=True)
        # num_features = self.backbone.fc.in_features
        # self.backbone.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(num_features, num_classes)
        #     )
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass
    
    def freeze_backbone(self):
        # TODO: Freeze backbone, unfreeze head
        pass
    
    def unfreeze_top_layers(self, num_layers=2):
        # TODO: Unfreeze top layers for fine-tuning
        pass

# ============================================================================
# STEP 5: Training (Copy from model_resnet50.py and modify model name)
# ============================================================================
def train_model():
    set_seeds()
    device = get_device()
    
    # Load data (uses shared utilities - same splits for all models)
    image_paths, labels = load_dataset_paths()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(
        image_paths, labels
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Your model
    model = YourModelClassifier(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    model.freeze_backbone()
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Phase 1: Train head
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
            torch.save(model.state_dict(), 'yourmodel_phase1_best.pt')
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break
    
    # Phase 2: Fine-tuning (similar structure)
    # ... (copy from model_resnet50.py)
    
    # Evaluation
    results = evaluate_model(model, test_loader, criterion, device)
    
    # Save results
    import json
    with open('yourmodel_results.json', 'w') as f:
        json.dump({
            'test_acc': float(results['test_acc']),
            'macro_f1': float(results['macro_f1']),
            'weighted_f1': float(results['weighted_f1']),
            'mcc': float(results['mcc']),
            'per_class_f1': results['per_class_f1'].tolist()
        }, f, indent=2)
    
    return model, results, history

# ============================================================================
# STEP 6: Run Training
# ============================================================================
if __name__ == '__main__':
    model, results, history = train_model()

