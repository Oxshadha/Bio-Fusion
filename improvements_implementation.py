"""
Implementation of Key Improvements for Model Training
1. Stain Normalization
2. Focal Loss
3. Class-Specific Loss Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ============================================================================
# 1. STAIN NORMALIZATION
# ============================================================================

def get_stain_normalized_transform():
    """
    Get transform with stain normalization
    Requires: !pip install torchstain
    """
    try:
        from torchstain import MacenkoNormalizer
        from torchvision import transforms
        
        # Create normalizer
        normalizer = MacenkoNormalizer(backend='torch')
        
        def normalize_stain(image):
            """Normalize H&E stain"""
            # Convert PIL to tensor
            if isinstance(image, Image.Image):
                image_tensor = transforms.ToTensor()(image)
            else:
                image_tensor = image
            
            # Normalize (expects [C, H, W] format)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            normalized = normalizer.normalize(image_tensor)
            
            # Convert back to PIL if needed
            if isinstance(image, Image.Image):
                normalized = normalized.squeeze(0)
                normalized = transforms.ToPILImage()(normalized)
            
            return normalized
        
        # Create transform pipeline
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Reduced from 90
            # Stain normalization BEFORE other transforms
            transforms.Lambda(normalize_stain),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Lambda(normalize_stain),  # Also normalize val/test
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_test_transform
    
    except ImportError:
        print("Warning: torchstain not installed. Using standard transforms.")
        print("Install with: !pip install torchstain")
        from shared_utilities import get_transforms
        return get_transforms(augment=True), get_transforms(augment=False)

# ============================================================================
# 2. FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum'
    """
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

# ============================================================================
# 3. CLASS-SPECIFIC LOSS WEIGHTING
# ============================================================================

def get_class_weights_for_confusion(y_train, confusion_pairs=None):
    """
    Create class weights that penalize specific confusions
    
    Args:
        y_train: Training labels
        confusion_pairs: List of (class1, class2) pairs to penalize more
                        e.g., [(NOR_idx, TUM_idx), (MUC_idx, ADI_idx)]
    
    Returns:
        class_weights: Tensor of weights for each class
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Base weights (inverse frequency)
    base_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Increase weights for classes involved in critical confusions
    if confusion_pairs:
        for class1_idx, class2_idx in confusion_pairs:
            # Increase weight for both classes in confusion pair
            base_weights[class1_idx] *= 1.5
            base_weights[class2_idx] *= 1.5
    
    return torch.FloatTensor(base_weights)

# ============================================================================
# 4. COMBINED LOSS (Focal + Class Weights)
# ============================================================================

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights
    Combines benefits of both approaches
    """
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

# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

"""
# In your training script:

# 1. Use stain normalization
train_transform, val_test_transform = get_stain_normalized_transform()
train_dataset = HistopathologyDataset(X_train, y_train, transform=train_transform)
# ... etc

# 2. Use Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# OR use Weighted Focal Loss for critical confusions
confusion_pairs = [
    (CLASS_TO_IDX['NOR'], CLASS_TO_IDX['TUM']),  # Penalize NOR-TUM confusion
    (CLASS_TO_IDX['MUC'], CLASS_TO_IDX['ADI'])   # Penalize MUC-ADI confusion
]
class_weights = get_class_weights_for_confusion(y_train, confusion_pairs)
criterion = WeightedFocalLoss(class_weights, alpha=0.25, gamma=2.0)

# 3. Train as usual
# ... training loop stays the same
"""

