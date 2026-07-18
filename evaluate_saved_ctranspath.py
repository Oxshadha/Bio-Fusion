"""
Evaluate and Save CTransPath Model
Run this in a new Colab session to evaluate the saved model and save to Google Drive
"""

# Setup
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
%pip install timm

# Upload files
from google.colab import files
files.upload()  # Upload: shared_utilities.py, model_ctranspath.py

import sys
sys.path.append('/content')
from shared_utilities import *
from model_ctranspath import CTransPathClassifier
import torch

# ============================================================================
# LOAD SAVED MODEL
# ============================================================================

print("="*60)
print("LOADING SAVED CTRANSPATH MODEL")
print("="*60)

device = get_device()

# Initialize model
model = CTransPathClassifier(num_classes=NUM_CLASSES, dropout=0.5).to(device)

# Try to load from local first, then Google Drive
local_path = 'ctranspath_final.pt'
drive_path = '/content/drive/MyDrive/BioFusion_Models/ctranspath_final.pt'

if os.path.exists(local_path):
    model.load_state_dict(torch.load(local_path, map_location=device))
    print(f"✓ Model loaded from local: {local_path}")
elif os.path.exists(drive_path):
    model.load_state_dict(torch.load(drive_path, map_location=device))
    print(f"✓ Model loaded from Google Drive: {drive_path}")
else:
    print("⚠ Model file not found!")
    print("  Looking for: ctranspath_final.pt")
    print("  Please ensure the model was saved during training")

# ============================================================================
# LOAD DATA AND EVALUATE
# ============================================================================

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

# Load data (same splits as training)
image_paths, labels = load_dataset_paths()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(
    image_paths, labels
)
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test
)

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

from shared_utilities import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Evaluate on test set
results = evaluate_model(model, test_loader, criterion, device)

print("\n" + "="*60)
print("CTRANSPATH FINAL RESULTS")
print("="*60)
print(f"Test Accuracy: {results['test_acc']:.2f}%")
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"Weighted F1: {results['weighted_f1']:.4f}")
print(f"MCC: {results['mcc']:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Confusion matrix
plot_confusion_matrix(
    results['labels'], results['predictions'], 
    'CTransPath', save_path='ctranspath_cm.png'
)
print("✓ Confusion matrix saved: ctranspath_cm.png")

# t-SNE
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
        print("✓ t-SNE plot saved: ctranspath_tsne.png")
except Exception as e:
    print(f"⚠ Could not generate t-SNE: {e}")

# ============================================================================
# SAVE TO GOOGLE DRIVE
# ============================================================================

print("\n" + "="*60)
print("SAVING TO GOOGLE DRIVE")
print("="*60)

# Save model and results
save_model_and_results(model, results, 'ctranspath', save_to_drive=True)

print("\n" + "="*60)
print("✓ COMPLETE!")
print("="*60)
print("Model and results saved to Google Drive")
print("Location: /content/drive/MyDrive/BioFusion_Models/")
print("  - ctranspath_final.pt")
print("  - ctranspath_results.json")

