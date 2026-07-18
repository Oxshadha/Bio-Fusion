# Research Paper Techniques Added to Notebooks

## Summary

Both `ModelCtranspath.ipynb` and `Model ensemble.ipynb` have been updated to include techniques from the research paper "Pneumonia Detection and Lung Disease Assessment from Chest X-rays" (Jayawardena et al., 2025) and histopathology-specific preprocessing methods.

## Techniques Added

### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Status**: ✅ **Already Implemented and Active**

**Mathematical Foundation**:
- Divides image into small tiles (typically 8×8)
- Applies histogram equalization to each tile with a contrast limit
- Uses bilinear interpolation to combine tiles smoothly
- Formula: For each tile, the histogram is clipped at a threshold (clip limit) and redistributed

**Reference**: 
- Jayawardena et al. (2025) - Demonstrated effectiveness in medical imaging
- Sasi & Jayasree (2013) - CLAHE for myocardial perfusion images

**Location in Code**: `shared_utilities.py` → `CLAHETransform` class

### 2. Stain Normalization (Mathematical Methods)

**Status**: ⚠️ **Documented and Available, But Not Currently Active**

**Three Mathematical Approaches Documented**:

#### A. Macenko Normalization (SVD-based)
- **Mathematical Basis**: Singular Value Decomposition (SVD)
- **Process**:
  1. Convert RGB to Optical Density (OD) space: `OD = -log(I/I₀)`
  2. Remove background pixels (OD < threshold)
  3. Apply SVD: `OD = U × Σ × V^T`
  4. Extract stain vectors from first two principal components
  5. Project all pixels onto reference stain vectors
- **Advantage**: Fast, robust, widely used in histopathology

#### B. Vahadane Normalization (NMF-based)
- **Mathematical Basis**: Sparse Non-Negative Matrix Factorization (NMF)
- **Process**:
  1. Convert to OD space
  2. Decompose: `OD ≈ W × H`, where W contains stain vectors, H contains concentrations
  3. Enforce sparsity constraint on H
  4. Project onto reference stain vectors
- **Advantage**: More accurate for complex staining patterns

#### C. Reinhard Normalization (Color Space Method)
- **Mathematical Basis**: Color space transformation (RGB → LAB)
- **Process**:
  1. Convert source and target to LAB color space
  2. Compute statistics: `μ_s, σ_s` (source), `μ_t, σ_t` (target)
  3. Apply transformation: `L' = (L - μ_s) × (σ_t/σ_s) + μ_t`
  4. Convert back to RGB
- **Advantage**: Simple, fast, preserves overall appearance

**Implementation Code Provided**: Both notebooks include example code showing how to add Macenko normalization using `torchstain` library.

**Why It Matters**: Reduces color-based confusion (e.g., MUC ↔ ADI) caused by H&E staining variations between labs/batches.

### 3. Data Augmentation

**Status**: ✅ **Active (Reduced Intensity)**

**Techniques Applied**:
- Geometric: Random horizontal/vertical flips, rotation (±15°)
- Color: Subtle ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

**Rationale**: Simulates stain variations and tissue orientation differences without over-augmentation (as research paper suggests).

### 4. Transfer Learning

**Status**: ✅ **Active**

**Research Paper Insight**: Jayawardena et al. (2025) demonstrated that transfer learning with pretrained models significantly improves performance in medical imaging tasks.

**Applied In**:
- CTransPath notebook: CTransPath (histopathology-pretrained) or ConvNeXt-Base (ImageNet-pretrained)
- Hybrid notebook: EfficientNet-B4 + ViT-Base (both ImageNet-pretrained)

### 5. Ensemble Methods

**Status**: ✅ **Active (Hybrid Model Only)**

**Research Paper Insight**: The paper showed that combining multiple architectures improves robustness and accuracy.

**Applied In**: Hybrid Ensemble model combining EfficientNet (CNN) and ViT (Transformer) with attention-based fusion.

### 6. Focal Loss

**Status**: ✅ **Active**

**Purpose**: Addresses class imbalance and hard examples (gamma=2.0, alpha=0.25, label_smoothing=0.1)

### 7. Regularization Techniques

**Status**: ✅ **Active**

- Dropout (0.6)
- Weight Decay (1e-4)
- Label Smoothing (0.1)

## Where These Are Documented in Notebooks

### ModelCtranspath.ipynb:
1. **Section 4.1.1**: Image Preprocessing & Enhancement Techniques
   - Detailed mathematical explanations of CLAHE and stain normalization methods
   - Research paper references
2. **Section 4.1.2**: Stain Normalization Implementation (optional code)
3. **Cell after Section 4.1.2**: Python code demonstrating preprocessing techniques
4. **Section 3.B**: Research paper insight on transfer learning
5. **Section 10 (Conclusion)**: Research Paper Techniques Applied + References

### Model ensemble.ipynb:
1. **Section 4.1.1**: Image Preprocessing & Enhancement Techniques
   - Detailed mathematical explanations of CLAHE and stain normalization methods
   - Research paper references
   - Ensemble-specific benefits
2. **Section 4.1.2**: Stain Normalization Implementation (optional code)
3. **Cell after Section 4.1.2**: Python code demonstrating preprocessing techniques
4. **Section 3.B**: Research paper insight on transfer learning and ensemble methods
5. **Section 10 (Conclusion)**: Research Paper Techniques Applied + References

## Mathematical Formulas Documented

1. **CLAHE**: `H_clipped = min(H, clip_limit)`, then redistribution
2. **Macenko**: `OD = -log(I/I₀)`, then `OD = U × Σ × V^T`
3. **Vahadane**: `OD ≈ W × H` with sparsity constraint
4. **Reinhard**: `L' = (L - μ_s) × (σ_t/σ_s) + μ_t`
5. **Attention Fusion** (Hybrid): `features = w_eff × eff_features + w_vit × vit_features`, where `[w_eff, w_vit] = softmax(MLP(concat))`

## References Added

Both notebooks now include a complete References section citing:
1. Jayawardena et al. (2025) - Main research paper
2. Sasi & Jayasree (2013) - CLAHE technique
3. Macenko et al. (2009) - Macenko stain normalization
4. Vahadane et al. (2016) - Vahadane stain normalization
5. Reinhard et al. (2001) - Reinhard color normalization

## Next Steps (Optional)

If you want to **activate stain normalization**:

1. Install library: `!pip install torchstain`
2. Uncomment the stain normalization code in the transform pipeline
3. Add `transforms.Lambda(normalize_stain)` to the transform list

**Note**: Stain normalization adds computational overhead but can significantly reduce color-based confusion (especially MUC ↔ ADI).

---

**All techniques from the research paper have been documented and integrated into both notebooks!** ✓

