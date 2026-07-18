# Model Saving and Sharing Guide

## Where Models Are Saved

### Automatic Saving (Both Locations):

1. **Local (Colab session)**: `/content/`
   - Files: `*_final.pt` (model weights)
   - Files: `*_results.json` (performance metrics)
   - **Note**: These are lost when Colab session ends

2. **Google Drive (Persistent)**: `/content/drive/MyDrive/BioFusion_Models/`
   - Files: `*_final.pt` (model weights)
   - Files: `*_results.json` (performance metrics)
   - **Persistent**: Saved permanently, accessible anytime

### Model Files Saved:

- `resnet50_final.pt` + `resnet50_results.json`
- `efficientnet_final.pt` + `efficientnet_results.json`
- `densenet_final.pt` + `densenet_results.json`
- `uni_final.pt` + `uni_results.json` (ViT-Base)
- `hybrid_final.pt` + `hybrid_results.json`
- `ctranspath_final.pt` + `ctranspath_results.json`

---

## For Judges: How to Run Your Notebook

### Option 1: Share Google Drive Folder (Recommended)

1. **Share Google Drive folder** with judges:
   - Folder: `BioFusion_Models/` in your Google Drive
   - Give read access to judges

2. **Judges can then**:
   - Mount their own Google Drive
   - Access your models from shared folder
   - Run inference easily

### Option 2: Download and Upload Models

1. **You download** models from Google Drive
2. **Upload to** GitHub, Google Drive (public), or submission form
3. **Judges download** and upload to their Colab

### Option 3: Include in Notebook (Small Models Only)

- For smaller models, you can save as base64 or include in notebook
- **Not recommended** for large models (files too big)

---

## Recommended Approach for Submission

### Step 1: Save All Models to Google Drive

All model scripts now automatically:
- ✅ Save locally (for immediate use)
- ✅ Save to Google Drive (for persistence)
- ✅ Save results JSON (for comparison)

### Step 2: Create Inference Notebook

Use `inference_notebook_template.ipynb` which:
- ✅ Loads models from Google Drive
- ✅ Runs inference on single images
- ✅ Runs inference on folders
- ✅ Easy for judges to use

### Step 3: Share with Judges

**Option A: Google Drive Sharing**
```
1. Share BioFusion_Models/ folder with judges
2. Judges mount their Google Drive
3. Models automatically accessible
```

**Option B: Download and Re-upload**
```
1. Download all *_final.pt files from Google Drive
2. Upload to submission form or GitHub
3. Judges download and use
```

---

## Inference Code for Judges

### Simple Inference Example:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Load model
import torch
from model_uni import UNIClassifier

model = UNIClassifier(num_classes=8, dropout=0.5)
model.load_state_dict(torch.load(
    '/content/drive/MyDrive/BioFusion_Models/uni_final.pt'
))
model = model.to('cuda')
model.eval()

# 3. Run inference
from PIL import Image
from shared_utilities import get_transforms

transform = get_transforms(augment=False)
image = Image.open('test_image.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to('cuda')

with torch.no_grad():
    outputs = model(image_tensor)
    predicted = torch.argmax(outputs, dim=1).item()

print(f"Predicted: {CLASSES[predicted]}")
```

---

## File Structure for Submission

### Recommended Structure:

```
BioFusion_Submission/
├── model_resnet50.py
├── model_efficientnet.py
├── model_densenet.py
├── model_uni.py
├── model_ensemble_hybrid.py
├── model_ctranspath.py
├── shared_utilities.py
├── compare_models.py
├── inference_notebook_template.ipynb  ⭐ For judges
├── README.md
└── Models/  (or link to Google Drive)
    ├── resnet50_final.pt
    ├── efficientnet_final.pt
    ├── densenet_final.pt
    ├── uni_final.pt
    ├── hybrid_final.pt
    └── ctranspath_final.pt
```

---

## Best Practice for Hackathon Submission

### 1. **Save to Google Drive** (Automatic)
- All models automatically saved
- Persistent storage
- Easy to share

### 2. **Create Inference Notebook**
- Use `inference_notebook_template.ipynb`
- Judges can run easily
- Includes examples

### 3. **Document Model Locations**
- In README, specify where models are saved
- Provide Google Drive link (if sharing)
- Or provide download instructions

### 4. **Test Inference Notebook**
- Before submission, test that judges can:
  - Load models
  - Run inference
  - See results

---

## Quick Checklist

Before submission, ensure:

- [ ] All models saved to Google Drive
- [ ] All `*_results.json` files saved
- [ ] Inference notebook created and tested
- [ ] README explains how to load models
- [ ] Models accessible to judges (Drive share or download)

---

## Example: Sharing Google Drive Folder

1. **Go to Google Drive**
2. **Right-click** `BioFusion_Models/` folder
3. **Share** → Add judges' emails
4. **Give "Viewer" access**
5. **Judges can access** models from their Colab

---

**All model scripts now automatically save to Google Drive!** ✅

