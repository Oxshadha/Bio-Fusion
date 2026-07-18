# Multi-Model Training Guide for Google Colab

## Overview

This guide explains how to train 4 different models simultaneously using 4 Google Colab accounts, then compare their performance.

## ✅ Is Training 4 Models Good or Bad?

### **GOOD** ✅
- **Ensemble Potential**: Can combine predictions for better accuracy
- **Architecture Comparison**: See which model works best for histopathology
- **Parallel Training**: Saves time (4x faster than sequential)
- **Robust Evaluation**: Multiple models validate findings

### **Potential Issues** ⚠️
- **Variable Control**: Hard to isolate what causes improvements
- **Solution**: Use **shared utilities** - same data splits, transforms, training loop
- Only change: **model architecture**

## Architecture Choices

1. **ResNet50** (Baseline) - Standard, proven
2. **EfficientNet-B4** - Better efficiency, good performance
3. **DenseNet121** - Better feature preservation, good for textures
4. **UNI** (Universal Medical Image Representation) - Medical vision transformer pretrained on large-scale medical imaging data

## Setup Instructions

### Step 1: Upload Shared Utilities

In **each Colab account**, upload `shared_utilities.py`:

```python
from google.colab import files
files.upload()  # Upload shared_utilities.py
```

### Step 2: Install Dependencies (if needed)

```python
# For EfficientNet
!pip install efficientnet-pytorch

# For other models, install as needed
```

### Step 3: Run Model-Specific Training

**Colab Account 1**: Run `model_resnet50.py`
**Colab Account 2**: Run `model_efficientnet.py`
**Colab Account 3**: Run `model_densenet.py`
**Colab Account 4**: Run `model_uni.py` (UNI pretrained model)

### Step 4: Download Results

From each Colab, download:
- `*_results.json` (performance metrics)
  - `resnet50_results.json`
  - `efficientnet_results.json`
  - `densenet_results.json`
  - `uni_results.json`
- `*_cm.png` (confusion matrix, optional)
- `*_final.pt` (model weights, optional)

### Step 5: Compare Models

Run `compare_models.py` with all result files to generate:
- Overall performance comparison
- Per-class F1 comparison
- Best model identification

## File Structure

```
BioFusion/
├── shared_utilities.py      # Common code (data loading, training, eval)
├── model_resnet50.py         # ResNet50 training script
├── model_efficientnet.py    # EfficientNet training script
├── model_densenet.py        # DenseNet training script
├── model_[your_model].py    # 4th model script
├── compare_models.py        # Comparison script
└── MULTI_MODEL_GUIDE.md     # This file
```

## Key Features of Shared Utilities

✅ **No Code Duplication**: 
- Dataset loading (once)
- Data splits (same for all models)
- Training loop (same logic)
- Evaluation (same metrics)

✅ **Consistent Comparison**:
- Same random seed
- Same train/val/test splits
- Same transforms
- Same hyperparameters (except model architecture)

✅ **Easy to Extend**:
- Add new model = copy template, change architecture
- All evaluation/visualization code already there

## Expected Results

After training all 4 models, you'll have:

1. **Performance Metrics**:
   - Test accuracy for each model
   - Macro F1, Weighted F1, MCC
   - Per-class F1 scores

2. **Visualizations**:
   - Confusion matrix for each model
   - Comparison plots
   - Per-class F1 comparison

3. **Best Model Identification**:
   - Which architecture performs best
   - Which classes each model handles well

## Tips for Success

1. **Use Same Random Seed**: All models use `RANDOM_SEED=42` for fair comparison
2. **Same Data Splits**: `create_splits()` ensures identical train/val/test
3. **Monitor Training**: Check each Colab periodically
4. **Save Results**: Download `*_results.json` files immediately after training
5. **Compare Fairly**: All models use same hyperparameters (LR, batch size, etc.)

## Troubleshooting

**Issue**: Models show very different results
- **Check**: Are you using the same data splits? (Use `shared_utilities.create_splits()`)
- **Check**: Are transforms the same? (Use `shared_utilities.get_transforms()`)

**Issue**: One model trains much slower
- **Normal**: Different architectures have different speeds
- EfficientNet might be slower but more accurate
- DenseNet might be faster but less accurate

**Issue**: Results files not found
- **Check**: Did you download `*_results.json` from each Colab?
- **Check**: Are files in the same directory as `compare_models.py`?

## Next Steps After Comparison

1. **Identify Best Model**: Use `compare_models.py` output
2. **Analyze Per-Class Performance**: See which model handles which classes best
3. **Consider Ensemble**: Combine predictions from top 2-3 models
4. **Further Tuning**: Focus on best model for hyperparameter tuning

## Ensemble Approach (Optional)

If you want to combine models:

```python
# Simple voting ensemble
predictions_model1 = model1(test_images)
predictions_model2 = model2(test_images)
predictions_model3 = model3(test_images)

# Average probabilities
ensemble_predictions = (predictions_model1 + 
                       predictions_model2 + 
                       predictions_model3) / 3
```

This often improves accuracy by 2-5% over the best single model.

---

**Good luck with your multi-model training!** 🚀

