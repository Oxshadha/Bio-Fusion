# ViT Model Training - Google Colab Setup Instructions

This guide explains how to run the Vision Transformer (ViT) model training in Google Colab with Kaggle dataset integration.

## Prerequisites

1. **Kaggle Account**: Sign up at https://www.kaggle.com/
2. **Kaggle API Token**: 
   - Go to your Kaggle account settings
   - Scroll to "API" section
   - Click "Create New Token"
   - Download `kaggle.json` file
3. **Google Colab**: Access at https://colab.research.google.com/
4. **Shared Utilities File**: You need `shared_utilities.py` file

## Step-by-Step Setup

### Step 1: Upload Files to Colab

1. Open a new Google Colab notebook
2. Upload `shared_utilities.py` to Colab:
   - Click on folder icon (📁) in left sidebar
   - Click "Upload to session storage"
   - Upload `shared_utilities.py`

### Step 2: Copy Code Cells

Open `VIT_model_colab.py` and copy each cell (marked with `# CELL X:`) into separate cells in your Colab notebook.

**Important**: Each `# CELL X:` section should be in its own Colab cell!

### Step 3: Run Cells in Order

Run cells sequentially from top to bottom:

#### Cell 1: Setup Kaggle and Download Dataset
- This will prompt you to upload `kaggle.json`
- Downloads and extracts the dataset automatically

#### Cell 2: Install Required Packages
- Installs all necessary Python packages

#### Cell 3: Mount Google Drive (Optional)
- Mounts your Google Drive to save models
- Creates model directory: `/content/drive/MyDrive/BioFusion_Models`

#### Cell 4: Import Libraries
- Imports all required libraries
- Imports from `shared_utilities.py`

#### Cell 5: Improvements Implementation
- Defines stain normalization, focal loss, and class weighting

#### Cell 6: ViT Model Definition
- Defines the Vision Transformer model class

#### Cell 7: Configuration and Setup
- Sets hyperparameters (20 epochs: 10+10)
- Verifies GPU availability

#### Cell 8: Load Dataset and Create Splits
- Loads dataset from Kaggle download
- Creates train/val/test splits

#### Cell 9: Setup Data Loaders
- Creates data loaders with stain normalization

#### Cell 10: Setup Loss Function
- Configures weighted focal loss with class weighting

#### Cell 11: Initialize Model
- Initializes ViT model and moves to GPU

#### Cell 12: Phase 1 Training
- Trains classifier head (10 epochs)
- Backbone frozen

#### Cell 13: Phase 2 Training
- Fine-tunes transformer layers (10 epochs)
- Lower learning rate

#### Cell 14: Final Evaluation
- Evaluates on test set
- Shows detailed metrics

#### Cell 15: Visualizations
- Creates confusion matrix
- Creates t-SNE visualization

#### Cell 16: Save Model and Results
- Saves model weights
- Saves results to local and Google Drive

#### Cell 17: Training History Plot
- Plots training curves

## Features Included

✅ **Stain Normalization** (Macenko method)  
✅ **Weighted Focal Loss** with class weighting  
✅ **20 Epochs** (10 Phase 1 + 10 Phase 2)  
✅ **Kaggle Dataset Integration**  
✅ **Google Drive Model Saving**  
✅ **GPU Support** (automatic detection)  
✅ **Early Stopping**  
✅ **Comprehensive Visualizations**  

## Expected Outputs

After training completes, you'll have:

1. **Model Files**:
   - `vit_phase1_best.pt` - Best model from Phase 1
   - `vit_final.pt` - Final trained model
   - Saved to both local and Google Drive

2. **Results File**:
   - `vit_results.json` - All metrics and training history
   - Saved to both local and Google Drive

3. **Visualizations**:
   - `vit_cm.png` - Confusion matrix
   - `vit_tsne.png` - t-SNE visualization
   - `vit_training_history.png` - Training curves

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'shared_utilities'"
**Solution**: Make sure `shared_utilities.py` is uploaded to Colab and in the `/content` directory

### Issue: "Kaggle API error"
**Solution**: 
- Verify `kaggle.json` is uploaded correctly
- Check that your Kaggle account has access to the dataset
- Ensure API token is not expired

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Use gradient accumulation if needed

### Issue: "Dataset not found"
**Solution**: 
- Make sure Cell 1 completed successfully
- Check that dataset path is correct: `/content/GCHTID/HMU-GC-HE-30K/all_image`

## Notes

- **GPU Runtime**: Colab provides free GPU (T4). Make sure to select "GPU" in Runtime → Change runtime type
- **Session Timeout**: Colab sessions timeout after inactivity. Save your work frequently!
- **Model Saving**: Models are saved to Google Drive, so they persist after session ends
- **Training Time**: With GPU, expect ~2-4 hours for full training (20 epochs)

## Quick Start Checklist

- [ ] Upload `shared_utilities.py` to Colab
- [ ] Upload `kaggle.json` when prompted in Cell 1
- [ ] Select GPU runtime (Runtime → Change runtime type → GPU)
- [ ] Copy all cells from `VIT_model_colab.py` to Colab
- [ ] Run cells sequentially
- [ ] Check Google Drive for saved models

## Support

If you encounter issues:
1. Check that all cells ran successfully
2. Verify GPU is enabled (Runtime → Change runtime type)
3. Check Colab logs for error messages
4. Ensure all required files are uploaded

Good luck with your training! 🚀

