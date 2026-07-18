"""
Ensemble ViT-Base and EfficientNet-B4 Models
Expected improvement: 67.44% → 69-71%
"""

import torch
import torch.nn as nn
import json
import numpy as np


# Load saved model results
def load_model_results():
    """Load results from all trained models"""
    results = {}
    try:
        with open('resnet50_results.json', 'r') as f:
            results['ResNet50'] = json.load(f)
    except:
        print("ResNet50 results not found")
    
    try:
        with open('efficientnet_results.json', 'r') as f:
            results['EfficientNet-B4'] = json.load(f)
    except:
        print("EfficientNet results not found")
    
    try:
        with open('densenet_results.json', 'r') as f:
            results['DenseNet121'] = json.load(f)
    except:
        print("DenseNet results not found")
    
    try:
        with open('uni_results.json', 'r') as f:
            results['ViT-Base'] = json.load(f)
    except:
        print("ViT-Base results not found")
    
    return results

def ensemble_predictions(models_dict, test_loader, device, weights=None):
    """
    Create ensemble predictions from multiple models
    
    Args:
        models_dict: Dict of {model_name: model_object}
        test_loader: Test data loader
        device: Device to run on
        weights: Optional weights for each model (default: equal)
    
    Returns:
        ensemble_predictions, true_labels
    """
    if weights is None:
        weights = {name: 1.0 for name in models_dict.keys()}
        # Normalize weights
        total = sum(weights.values())
        weights = {name: w/total for name, w in weights.items()}
    
    all_predictions = {name: [] for name in models_dict.keys()}
    true_labels = []
    
    for name, model in models_dict.items():
        model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            true_labels.extend(labels.numpy())
            
            # Get predictions from each model
            for name, model in models_dict.items():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_predictions[name].append(probs.cpu().numpy())
    
    # Stack predictions
    for name in all_predictions.keys():
        all_predictions[name] = np.vstack(all_predictions[name])
    
    # Weighted ensemble
    ensemble_probs = None
    for name, probs in all_predictions.items():
        weighted_probs = probs * weights[name]
        if ensemble_probs is None:
            ensemble_probs = weighted_probs
        else:
            ensemble_probs += weighted_probs
    
    # Get final predictions
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    true_labels = np.array(true_labels)
    
    return ensemble_preds, true_labels

def evaluate_ensemble(predictions, labels, model_name="Ensemble"):
    """Evaluate ensemble performance"""
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        classification_report, confusion_matrix
    )
    
    acc = accuracy_score(labels, predictions) * 100
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    mcc = matthews_corrcoef(labels, predictions)
    
    print("="*60)
    print(f"{model_name.upper()} RESULTS")
    print("="*60)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    # Classification report
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)
    print(classification_report(
        labels, predictions,
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix
    plot_confusion_matrix(
        labels, predictions,
        model_name, save_path=f'{model_name.lower().replace(" ", "_")}_cm.png'
    )
    
    return {
        'test_acc': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mcc': mcc,
        'predictions': predictions,
        'labels': labels
    }

# Example usage:
"""
# Load models (you need to load the actual model objects)
vit_model = load_vit_model('uni_final.pt')
eff_model = load_efficientnet_model('efficientnet_final.pt')

models_dict = {
    'ViT-Base': vit_model,
    'EfficientNet-B4': eff_model
}

# Equal weights
ensemble_preds, true_labels = ensemble_predictions(
    models_dict, test_loader, device
)

# Or weighted (give more weight to better model)
weights = {
    'ViT-Base': 0.6,  # Better model gets more weight
    'EfficientNet-B4': 0.4
}
ensemble_preds, true_labels = ensemble_predictions(
    models_dict, test_loader, device, weights=weights
)

results = evaluate_ensemble(ensemble_preds, true_labels, "ViT+EfficientNet Ensemble")
"""

