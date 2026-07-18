# UNI Model Setup Notes

## About UNI

**UNI (Universal Medical Image Representation)** is a medical vision transformer model pretrained on large-scale medical imaging data, including:
- Histopathology images
- Radiology images
- Other medical imaging modalities

This makes it particularly suitable for histopathology classification tasks.

## Installation Options

UNI may be available through different sources:

### Option 1: Hugging Face (Preferred)
```python
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained('microsoft/uni')
processor = AutoImageProcessor.from_pretrained('microsoft/uni')
```

### Option 2: Timm Library
```python
import timm
model = timm.create_model('uni_vit_base', pretrained=True)
```

### Option 3: Official UNI Repository
If UNI has an official repository, you may need to:
```bash
!pip install uni-medical
# or
!git clone https://github.com/microsoft/uni
```

## Model Architecture

UNI is typically a Vision Transformer (ViT) architecture:
- **Input**: 224×224 images (same as other models)
- **Backbone**: Transformer encoder with medical pretraining
- **Output**: Feature embeddings (768 or 1024 dimensions)
- **Classifier**: Custom head for 8-class classification

## Key Differences from CNN Models

1. **Attention Mechanism**: UNI uses self-attention to capture long-range dependencies
2. **Medical Pretraining**: Pretrained specifically on medical images (not ImageNet)
3. **Patch-based Processing**: Divides images into patches (like standard ViT)

## Expected Advantages

1. **Domain-Specific**: Pretrained on medical data, should generalize better
2. **Attention**: Can focus on clinically relevant regions
3. **Transfer Learning**: Better starting point than ImageNet-pretrained models

## Troubleshooting

### Issue: "UNI model not found"
**Solution**: The script includes a fallback to standard ViT-Base from timm, which will still work well.

### Issue: "Input format mismatch"
**Solution**: UNI might expect different input format. Check if you need:
- Different image preprocessing
- Different input tensor shape
- Specific tokenization

### Issue: "Out of memory"
**Solution**: 
- Reduce batch size to 16 or 8
- Use gradient checkpointing
- Use smaller UNI variant if available

## Model Variants

If available, you might want to try:
- **UNI-Base**: Standard size (recommended)
- **UNI-Large**: Larger capacity (if memory allows)
- **UNI-Small**: Faster training (if time constrained)

## Fine-Tuning Strategy

For UNI, we use:
- **Phase 1**: Freeze all transformer layers, train classifier only
- **Phase 2**: Unfreeze last 2 transformer blocks, fine-tune with lower LR (1e-5)

This is more conservative than CNN models because:
- Transformers have more parameters
- Medical pretraining is already domain-specific
- Lower learning rate prevents catastrophic forgetting

## Expected Performance

UNI should perform **better than ImageNet-pretrained models** because:
- ✅ Pretrained on medical/histopathology data
- ✅ Better domain alignment
- ✅ Attention mechanism for spatial reasoning

**Expected**: 65-75% accuracy (better than ResNet50's 60.6%)

## Alternative: If UNI Not Available

The script includes a fallback to **ViT-Base** from timm, which:
- Still uses transformer architecture
- Pretrained on ImageNet (not medical, but still good)
- Will work with the same code structure

## References

If UNI has specific installation instructions, check:
- Microsoft Research publications
- Hugging Face model hub
- Official UNI repository

---

**Note**: The script is designed to work even if UNI is not directly available, falling back to standard ViT which is still a strong baseline.

