# Image Classification with Random Forest and SVM

Classifying images into five categories using classical machine learning on raw pixel features, with a direct comparison of tree-based and kernel-based approaches.

**Author:** Ayleen Santander
**Course:** Willis College — Business Intelligence and Data Analytics, Module 11

---

## Business Question

Can a classical machine learning model reliably sort product or content images into categories without the cost and complexity of deep learning infrastructure?

This matters because CNN-based image classification requires GPUs, large labelled datasets, and specialist expertise. If a Random Forest trained on a laptop can hit acceptable accuracy, it is the cheaper answer. This project tests where that ceiling actually sits.

---

## Dataset

309 images across five classes, drawn from a Caltech-101 subset.

| Class | Images |
|---|---|
| sunflower | 85 |
| dalmatian | 67 |
| soccer_ball | 54 |
| pizza | 52 |
| dollar_bill | 51 |

Mildly imbalanced, with sunflower at 1.7× the smallest class. Handled with stratified sampling at the split stage.

---

## Approach

**Preprocessing pipeline**

1. Load images from class-labelled folders
2. Convert to grayscale
3. Resize to a uniform 64×64
4. Flatten to 4,096 features per image
5. Normalize pixel values to a 0–1 range
6. Stratified 80/20 train/test split (247 train, 62 test)

Preprocessing was written as a single reusable function so that training and inference apply identical transformations — a common source of silent model failure when the two drift apart.

**Models**

- Random Forest, tuned via GridSearchCV across `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`
- Support Vector Machine (RBF and linear kernels), tuned across `C`, `gamma`, and `kernel`
- 3-fold cross-validation on the training set only; the test set was held out until final evaluation

---

## Results

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| **Random Forest (tuned)** | **64.5%** | 0.65 |
| Random Forest (default) | 67.7% | 0.68 |
| SVM (tuned) | 58.1% | 0.58 |

Against a 20% random baseline, all models learn real signal. None reach production quality.

Notably, the tuned Random Forest scored *below* the default configuration on the test set despite winning on cross-validation. With only 62 test images, a 3-point difference is roughly two images — well inside noise. This is a useful caution against over-reading small-sample results.

### Per-class performance (Random Forest)

| Class | Precision | Recall |
|---|---|---|
| dollar_bill | 0.86 | 0.60 |
| pizza | 0.75 | 0.60 |
| dalmatian | 0.73 | 0.57 |
| soccer_ball | 0.62 | 0.73 |
| sunflower | 0.52 | 0.71 |

---

## Key Findings

**1. Sunflower absorbs the errors.** As the largest class, it becomes the model's default guess under uncertainty — 11 of 22 misclassifications land there. Its precision (0.52) is the weakest of any class despite strong recall.

**2. Dalmatian and soccer_ball are systematically confused.** Both reduce to high-contrast black-and-white blobs at 64×64 grayscale. The model swaps them in both directions.

**3. The model is a center-pixel detector.** Feature importance analysis shows every one of the top 20 pixels clustered in rows 19–29, columns 24–35 — a small patch at the image center. Roughly 5% of the 4,096 features do meaningful work; the rest contribute near zero.

This is a fragile strategy. It works because photographers frame subjects centrally, and would break on off-center or zoomed-out images.

**4. The ceiling is the features, not the algorithm.** Three model configurations produced accuracies between 58% and 68%, and the same classes failed the same way in each. When changing algorithms only shuffles which errors appear, the limitation lies upstream in how the data is represented.

Flattened pixels carry no spatial information. Pixel 2,317 is treated as an independent column with no knowledge that it neighbours pixel 2,318, so the model cannot learn edges, shapes, or texture — only intensity at fixed positions.

---

## Recommendations

| Priority | Action | Rationale |
|---|---|---|
| High | Retain color (`convert('RGB')`) | Sunflower is yellow, pizza red-orange, dollar_bill green. Grayscale conversion discarded the strongest separating signal. |
| High | Use a CNN | Learns spatial features directly rather than treating pixels as independent columns. |
| Medium | Expand the dataset | 51–85 images per class is thin for 4,096 features. |
| Low | Apply PCA | With ~95% of features contributing near zero, dimensionality reduction to 50–100 components would likely match accuracy at a fraction of the compute. |

---

## Tools

Python · scikit-learn · NumPy · Pandas · Pillow · Matplotlib · Seaborn · Google Colab

---

## Repository Contents

```
M11_A11_Image_Classification_Random_Forest.ipynb   Full analysis notebook
README.md                                          This file
```

---

## What This Project Demonstrates

- End-to-end image preprocessing pipeline with training/inference parity
- Hyperparameter tuning via GridSearchCV with proper train/test isolation
- Multi-class evaluation using precision, recall, F1, and confusion matrix analysis
- Feature importance interpretation mapped back to spatial image coordinates
- Comparative model assessment with reasoning about *why* results converge
