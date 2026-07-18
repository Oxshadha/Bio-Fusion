"""
Model Comparison Script
Run this after all 4 models are trained
Downloads results from each Colab and compares them
"""

import json
import numpy as np
from shared_utilities import compare_models, plot_per_class_f1

# Load results from all models
# (Upload the *_results.json files from each Colab)

results_dict = {}

# Load ResNet50 results
try:
    with open('resnet50_results.json', 'r') as f:
        resnet50 = json.load(f)
        results_dict['ResNet50'] = resnet50
except FileNotFoundError:
    print("Warning: resnet50_results.json not found")

# Load EfficientNet results
try:
    with open('efficientnet_results.json', 'r') as f:
        efficientnet = json.load(f)
        results_dict['EfficientNet-B4'] = efficientnet
except FileNotFoundError:
    print("Warning: efficientnet_results.json not found")

# Load DenseNet results
try:
    with open('densenet_results.json', 'r') as f:
        densenet = json.load(f)
        results_dict['DenseNet121'] = densenet
except FileNotFoundError:
    print("Warning: densenet_results.json not found")

# Load UNI/ViT-Base results
try:
    with open('uni_results.json', 'r') as f:
        uni = json.load(f)
        results_dict['ViT-Base'] = uni
except FileNotFoundError:
    print("Warning: uni_results.json not found")

# Load Hybrid Ensemble results
try:
    with open('hybrid_results.json', 'r') as f:
        hybrid = json.load(f)
        results_dict['Hybrid (EfficientNet+ViT)'] = hybrid
except FileNotFoundError:
    print("Warning: hybrid_results.json not found")

# Load CTransPath results
try:
    with open('ctranspath_results.json', 'r') as f:
        ctranspath = json.load(f)
        results_dict['CTransPath'] = ctranspath
except FileNotFoundError:
    print("Warning: ctranspath_results.json not found")

if len(results_dict) > 0:
    # Compare models
    compare_models(results_dict, save_path='model_comparison.png')
    
    # Per-class F1 comparison
    per_class_f1_dict = {
        model: np.array(results['per_class_f1']) 
        for model, results in results_dict.items()
    }
    plot_per_class_f1(per_class_f1_dict, save_path='per_class_f1_comparison.png')
    
    # Find best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['test_acc']:.2f}%")
    print(f"   Macro F1: {best_model[1]['macro_f1']:.4f}")
else:
    print("No results found. Please upload *_results.json files first.")

