# Ethical AI Analysis & Explainability — Income Prediction Audit

**Module 11 · Assignment 14 — Introduction to AI**
Author: Ayleen Santander

A fairness and explainability audit of a machine learning model that predicts whether a person earns more than $50K per year. The point of the project is not the accuracy score — it is what the model does differently to different groups of people, and whether anyone can explain why.

---

## Business question

If this model were deployed to screen people — for credit, for a hiring pipeline, for a marketing offer — who would it help, who would it miss, and could the decision be defended to a regulator?

---

## Dataset

| | |
|---|---|
| Source | UCI Adult Census Income (Kaggle) |
| Rows | 32,561 |
| Features after encoding | 100 |
| Target | Income > $50K (1) vs ≤ $50K (0) |
| Class balance | 24,720 low / 7,841 high — **24.1% positive** |
| Sensitive attributes audited | `sex`, `race` (and both crossed) |

---

## Approach

1. **Prepare** — target separated *before* one-hot encoding to avoid leaking it into the feature set; stratified 80/20 split.
2. **Balance** — SMOTE applied to the training set only (26,048 → 39,550 rows, 50/50).
3. **Scale** — `StandardScaler` fitted on training data only.
4. **Model** — Logistic Regression (baseline) and Random Forest (200 trees).
5. **Audit fairness** — Fairlearn `MetricFrame`: selection rate, false positive rate, true positive rate, demographic parity and equalized odds.
6. **Explain** — SHAP beeswarm and mean |SHAP| for global drivers; waterfall plots for individual decisions.

---

## Model performance

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.689 | 0.629 | 0.658 | 0.889 |
| **Random Forest** | **0.846** | 0.694 | 0.647 | 0.669 | **0.895** |

**Accuracy is the wrong headline.** Always predicting "≤50K" already scores 75.9%. The model's real lift is about 9 points. ROC-AUC (0.89) and the minority-class F1 (0.67) are the numbers that matter.

**The dominant error is under-prediction.** The Random Forest misses 554 of 1,568 actual high earners (35.3% false negative rate) while wrongly flagging only 448 of 4,945 low earners (9.1% false positive rate). Despite balancing the training set 50/50, the model still flags only 22.4% of the test set against a true base rate of 24.1% — SMOTE barely moved the decision boundary at the default 0.5 threshold.

---

## Fairness findings

### By sex (Random Forest)

| | Accuracy | Selection rate | FPR | TPR (recall) | n |
|---|---|---|---|---|---|
| Female | 0.929 | 0.089 | 0.028 | 0.580 | 2,153 |
| Male | 0.805 | 0.292 | 0.130 | 0.659 | 4,360 |
| **Gap** | 0.124 | **0.203** | **0.103** | 0.079 | |

- **Demographic parity difference: 0.203** — men are flagged as high earners at 3.3× the rate of women.
- **Equalized odds difference: 0.103** — driven by the false positive rate gap.

### The accuracy gap runs the opposite way to how it looks

Female accuracy (0.929) appears better than male (0.805). It is not. The majority-class baseline is 89.0% for women and 69.5% for men, so the model's real lift is **+3.7 points for women against +10.9 points for men**. The model does substantially less useful work for women, and high accuracy hides it.

### By race

Selection rates range from 9.1% (Other) to 23.9% (White). Sample sizes for several groups are small (44–579 rows), so these figures are directional rather than conclusive — a point worth stating explicitly in any real audit.

---

## Explainability findings

**Global drivers (mean |SHAP|):** relationship status, age, marital status, education level, hours per week, capital gains.

**`fnlwgt` ranked #2 in the Random Forest's native feature importance — a red flag.** It is the census sampling weight, an estimate of how many people in the population a row represents. It is a survey design artifact with no causal link to anyone's income. It should be dropped, not modelled. Its high ranking is evidence that tree importance rewards high-cardinality noise.

**Local explanations:** SHAP waterfall plots decompose the single most confident ">50K" and "≤50K" predictions into per-feature contributions — the level of explanation needed to answer "why was I declined?"

---

## Recommendations

1. **Drop `fnlwgt`** and re-train. It adds no signal and inflates importance rankings.
2. **Tune the decision threshold**, do not leave it at 0.5. Threshold choice is a business decision (the cost of a missed high earner differs from the cost of a false flag) and it moves selection rates unevenly across groups — so it is a fairness lever as much as a performance lever.
3. **Reconsider SMOTE.** Interpolating across ~94 one-hot columns produces synthetic rows like "0.4 × Private + 0.6 × Self-employed" — people who do not exist. Class weighting is the safer alternative here.
4. **Report lift over baseline, not raw accuracy**, in every group-level table. Raw accuracy inverts the story.
5. **Apply a mitigation technique** (Fairlearn `ThresholdOptimizer` or `ExponentiatedGradient`) and re-measure, before any deployment discussion.

---

## Tech stack

`pandas` · `scikit-learn` · `imbalanced-learn` (SMOTE) · `Fairlearn` · `SHAP` · `LIME` · `matplotlib` · Google Colab

## Repository contents

```
M.11_A.14_Ethical_AI_Analysis_Explainability.ipynb   # full analysis
adult.csv                                            # dataset
README.md
```

## How to run

```bash
pip install pandas scikit-learn imbalanced-learn fairlearn shap lime matplotlib
```

Open the notebook in Google Colab or Jupyter, place `adult.csv` in the working directory, and run **Runtime → Restart and run all**.

---

## Caveat

The 0.10 disparity line used in the charts is a common rule of thumb, not a legal standard. Any real-world deployment would require legal review against the applicable jurisdiction's standard.
