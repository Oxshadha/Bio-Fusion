# Quick Start: 6 Models Training Guide

## Your 6 Models

| Colab Account | Model | File | Expected Accuracy |
|---------------|-------|------|-------------------|
| 1 | ResNet50 | `model_resnet50.py` | 60.84% ✅ |
| 2 | EfficientNet-B4 | `model_efficientnet.py` | 65.92% ✅ |
| 3 | DenseNet121 | `model_densenet.py` | 61.52% ✅ |
| 4 | ViT-Base | `model_uni.py` | 67.44% ✅ |
| 5 | **Hybrid (Eff+ViT)** | `model_ensemble_hybrid.py` | **69-72%** ⭐ |
| 6 | **CTransPath** | `model_ctranspath.py` | **70-75%** ⭐ |

---

## Setup for Each Colab Account

### Step 1: Upload Files
```python
from google.colab import files
files.upload()  # Upload:
# 1. shared_utilities.py (REQUIRED for all)
# 2. model_*.py (specific to each account)
```

### Step 2: Run Dataset Setup (Same for All)
```python
files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d orvile/gastric-cancer-histopathology-tissue-image-dataset
!unzip gastric-cancer-histopathology-tissue-image-dataset.zip -d GCHTID
```

### Step 3: Run Model Script
```python
# Just run the model_*.py file
# It will:
# - Load data using shared_utilities
# - Train the model
# - Save results to *_results.json
```

### Step 4: Download Results
- Download `*_results.json` from each Colab
- Download `*_cm.png` (confusion matrix, optional)
- Download `*_final.pt` (model weights, optional)

---

## Model 5: Hybrid Ensemble (EfficientNet + ViT)

### What It Does:
- Extracts features from **both** EfficientNet (CNN) and ViT (Transformer)
- **Concatenates** the features: `[eff_features, vit_features]`
- Trains a classifier on the combined features

### Why It Should Work Better:
- ✅ **CNN features**: Capture local texture patterns (important for histopathology)
- ✅ **Transformer features**: Capture global spatial relationships
- ✅ **Combined**: Best of both worlds

### Expected:
- **69-72% accuracy** (better than single models)
- Should handle both texture and spatial context

---

## Model 6: CTransPath

### What It Is:
- **Histopathology-specific** Vision Transformer
- Pretrained on **large-scale histopathology data** (not ImageNet!)
- Should be the **best single model**

### Why It Should Be Best:
- ✅ **Domain-specific pretraining**: Trained on histopathology images
- ✅ **Better feature alignment**: Features designed for medical imaging
- ✅ **Proven**: Used in many histopathology papers

### Expected:
- **70-75% accuracy** (best expected)
- Better than ImageNet-pretrained models

### If CTransPath Not Available:
- Script automatically uses **ConvNeXt-Base** or **ViT-Base**
- Still strong baselines
- **Script will work regardless**

---

## After All Models Trained

### Step 1: Collect Results
Download all `*_results.json` files:
- `resnet50_results.json`
- `efficientnet_results.json`
- `densenet_results.json`
- `uni_results.json` (ViT-Base)
- `hybrid_results.json` ⭐
- `ctranspath_results.json` ⭐

### Step 2: Compare All Models
```python
# Run compare_models.py with all result files
python compare_models.py
```

### Step 3: Create Final Ensemble (Optional)
```python
# Use ensemble_models.py
# Combine top 2-3 models
# Expected: 75-78% accuracy
```

---

## Expected Final Ranking

1. 🥇 **CTransPath**: 70-75% (domain-specific pretraining)
2. 🥈 **Hybrid Ensemble**: 69-72% (CNN + Transformer)
3. 🥉 **ViT-Base**: 67.44% (strong transformer)
4. EfficientNet-B4: 65.92% (efficient CNN)
5. DenseNet121: 61.52% (dense CNN)
6. ResNet50: 60.84% (baseline)

---

## Key Points

✅ **All models use `shared_utilities.py`** - ensures fair comparison
✅ **Same data splits** - identical train/val/test
✅ **Same transforms** - same preprocessing
✅ **Only architecture differs** - clear comparison

✅ **Hybrid combines CNN + Transformer** - best of both
✅ **CTransPath is domain-specific** - should be best

---

## Troubleshooting

### Hybrid Model Issues:
- **Memory**: May need smaller batch size (16 instead of 32)
- **Feature shapes**: Code handles different EfficientNet/ViT outputs

### CTransPath Issues:
- **Not found**: Script auto-fallback to ConvNeXt/ViT (still works!)
- **Installation**: See `CTRANSPATH_SETUP.md`

---

**Ready to train! Good luck!** 🚀

