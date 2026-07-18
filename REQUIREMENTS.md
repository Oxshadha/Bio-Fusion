# BioFusion Hackathon - Notebook and Report Requirements

## Overview
- **Submission Deadline**: 8 PM on 4th Jan 2026
- **Submission Format**: 
  - Notebook: `TeamName_Notebook.ipynb`
  - Report: `TeamName_Report.pdf`
- **Submission Method**: Google Form link (provided)

---

## A. Notebook Requirements (`.ipynb`)

### 1. Problem Definition
- [ ] Clinical/healthcare relevance
- [ ] What is being predicted?

### 2. Dataset Documentation
- [ ] Full citation of the dataset
- [ ] Variables/labels description
- [ ] Data distribution + basic analysis
- [ ] Preprocessing steps taken

### 3. Model Initialization & Pretraining Disclosure

#### A. Pretrained Model Used (if applicable)
- [ ] Model name
- [ ] Source
- [ ] Task it was originally trained on (e.g., ImageNet, clinical text, generic images)

#### B. Weight Usage
- [ ] Clearly state one of:
  - Used pretrained weights
  - Random initialization (training from scratch)

### 4. Model Development

#### Architecture & Design
- [ ] Full architecture (from scratch) or using a pretrained model
- [ ] Justification for the chosen design
- [ ] List and explain all hyperparameter choices:
  - Learning rate
  - Batch size
  - Optimizer
  - Other hyperparameters

#### Training Strategy
- [ ] Clearly show in code and markdown:
  - Which layers are frozen
  - Which layers are partially trainable
  - Which layers are fully trainable
  - Whether pretrained backbone is used only as a feature extractor or fully fine-tuned

#### For Deep Learning Models
**You must show how your model learns by displaying:**
- [ ] Forward pass → how inputs produce predictions
- [ ] Loss computation → how errors are calculated
- [ ] Backpropagation → how gradients are computed
- [ ] Optimizer update → how model weights are updated

**Important**: Do NOT hide training steps in one-line automated calls/high-level functions (e.g., `model.fit()`) without explanation.

#### For Machine Learning Models
- [ ] Clearly document your training procedure in code and markdown:
  - How the model was trained
  - Hyperparameters used
  - Data splitting strategy

#### Validation Approach
- [ ] Use train/validation/test split or k-fold cross-validation
- [ ] Explain your choice
- [ ] Show it in code/markdown

### 5. Outputs & Logs
- [ ] Training curves (loss, accuracy, etc.)
- [ ] Validation metrics
- [ ] Error analysis
- [ ] Computational constraints faced

### 6. Performance Metrics
- [ ] Primary metrics
- [ ] Secondary metrics
- [ ] Confusion matrix (if classification)
- [ ] ROC curves (if applicable)

### 7. Reproducibility
- [ ] Code cells must run when configurations are set up
- [ ] Random seeds must be set

### 8. Final Model File
- [ ] Include final model file

---

## B. Report Requirements (PDF)

### Format Specifications
- **Page Limit**: 5 pages
- **Font Size**: 12
- **Font Type**: Times New Roman
- **Line Spacing**: 1.15

### Required Sections

#### 1. Literature Review
- [ ] Minimum 3 research papers
- [ ] What has been done before?
- [ ] Gaps in existing work
- [ ] What your solution improves or proposes

#### 2. Problem Identification
- [ ] Who is affected?
- [ ] Why is this problem important?
- [ ] Specific unmet need in healthcare

#### 3. Dataset Justification
- [ ] Why the selected dataset is appropriate

#### 4. Methodology
- [ ] Data preprocessing pipeline
- [ ] Model architecture (diagrams encouraged)
- [ ] Training process
- [ ] Validation strategy

#### 5. Pretrained Model Usage & Adaptation (if applicable)

##### a. Rationale
- [ ] Why a pretrained model was chosen
- [ ] Why it is appropriate for the medical task

##### b. Modifications
- [ ] Architectural changes made (e.g., replaced classifier head)
- [ ] New layers added
- [ ] Output adaptation

##### c. Training Strategy
- [ ] Fine-tuning vs feature extraction
- [ ] Learning rates used for:
  - Pretrained layers
  - Newly added layers (if different)

##### d. Risk & Bias Discussion
- [ ] Domain mismatch (e.g., natural images → medical images)
- [ ] Potential bias inherited from pretraining data

#### 6. Results
- [ ] Metric tables
- [ ] Visualizations
- [ ] Error analysis
- [ ] Limitations of your model

#### 7. Real-world Application
- [ ] Proposed deployment scenario
- [ ] Potential users (clinicians, patients, hospitals)
- [ ] Integrating into healthcare workflow
- [ ] Risks & limitations

#### 8. Marketing & Impact Strategy
- [ ] Who would adopt it?
- [ ] Practical benefits
- [ ] Cost, accessibility, reach

#### 9. Future Improvements
- [ ] Model enhancements
- [ ] Additional data needs
- [ ] Clinical translation pathways

---

## Rules and Regulations

### Data Rules
- ✅ **Allowed**: Datasets from open source database repositories (e.g., Kaggle, OpenML)
- ✅ **Allowed**: Combining multiple datasets
- ✅ **Allowed**: Standard data augmentation techniques (rotation, flipping, cropping)
- ❌ **Not Allowed**: Datasets requiring subscriptions or membership
- ❌ **Not Allowed**: Adding samples from external datasets or scraping additional images/text
- ⚠️ **Requirement**: Dataset must have > 300 records (or clearly document justification for smaller datasets)

### Pretrained Components
- ✅ **Allowed**: Pretrained models, pretrained weights, and embeddings
- ⚠️ **Required**: Details of pretrained model and architecture must be clearly documented
- ⚠️ **Required**: Justification for choosing the specific model must be provided
- ⚠️ **Required**: Use of pretrained components must be explicitly declared

### Other Rules
- ❌ **Not Allowed**: AutoML tools
- ❌ **Not Allowed**: Previously built pipelines
- ❌ **Not Allowed**: Another person's work (plagiarism leads to disqualification)
- ⚠️ **Important**: Only 1 submission per team
- ⚠️ **Important**: Late submissions will not be accepted

### Allowed Resources
- ✅ Jupyter Notebook, Google Colab, Kaggle Notebook
- ✅ Publicly available Python packages (TensorFlow, PyTorch, scikit-learn, OpenCV, NumPy, Pandas, Matplotlib, etc.)
- ✅ Academic papers, online documentation, textbooks

---

## Evaluation Criteria

### A. Technical Evaluation (Model + Methods) — 65%

| Category | Weight | Focus Areas |
|----------|--------|-------------|
| **Model Design and Adaptation** | 20% | • Appropriateness of model choice for the medical problem<br>• If pretrained: correctness of adaptation<br>• If from scratch: soundness and originality of architecture<br>• Clear justification of design decisions |
| **Data Understanding & Preprocessing** | 10% | • Dataset suitability and understanding<br>• Handling of imbalance, noise, missing data<br>• Prevention of data leakage |
| **Training & Validation Strategy** | 10% | • Correct training loop implementation<br>• Hyperparameter choices and justification<br>• Validation method (train/val/test, k-fold) |
| **Performance Metrics** | 20% | • Correct choice of metrics for the task<br>• Honest reporting of results<br>• Interpretation of metrics<br>• Comparison with baselines (if provided) |
| **Error Analysis and Limitations** | 5% | • Insight into failures<br>• Model weaknesses |

### B. Notebook Quality — 20%

| Category | Weight | Focus Areas |
|----------|--------|-------------|
| **Code Clarity & Documentation** | 10% | • Readability, comments, logical cell order |
| **Reproducibility** | 10% | • Runs without errors<br>• Paths intact<br>• No hidden dependencies<br>• Explicit disclosure of pretrained components |

### C. Technical & Application Report — 15%

| Section | Weight | Focus Areas |
|---------|--------|-------------|
| **Literature Review & Problem Identification** | 5% | • Understanding of clinical context + research foundation |
| **Methodology Explanation** | 5% | • Clear reasoning, workflow diagrams, decisions justified<br>• Transparency of pretrained model usage<br>• Appropriateness of pretraining source<br>• Discussion of bias, domain mismatch, and limitations |
| **Results, Discussion, Real-World Application** | 5% | • Interpretation of findings<br>• Practicality<br>• Limitations |

---

## Quick Checklist Summary

### Before Submission
- [ ] Notebook renamed as `TeamName_Notebook.ipynb`
- [ ] Report renamed as `TeamName_Report.pdf`
- [ ] All code cells run without errors
- [ ] Random seeds set for reproducibility
- [ ] All required sections included in both notebook and report
- [ ] Pretrained models (if used) are clearly documented
- [ ] Dataset citation included
- [ ] Report format follows specifications (5 pages, Times New Roman 12pt, 1.15 spacing)
- [ ] Submission before 8 PM on 4th Jan 2026

### Key Reminders
- ⚠️ Show training steps explicitly (don't hide in `model.fit()`)
- ⚠️ Document all hyperparameters and justify choices
- ⚠️ Include error analysis and limitations
- ⚠️ Explain validation strategy clearly
- ⚠️ Discuss real-world application and impact

---

## Notes
- Projects are evaluated based on methodological rigor, transparency, and appropriateness of design
- Evaluation is the same whether models are trained from scratch or initialized with pretrained weights
- The judge's decision will be the final decision

