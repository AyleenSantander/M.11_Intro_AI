# Breast Cancer Diagnosis Classifier

Supervised machine learning pipeline that classifies breast tumours as **Benign** or **Malignant** from clinical and pathological features, with an interactive Gradio interface deployed on Hugging Face Spaces.

**Author:** Ayleen Santander
**Course:** Willis College Online — Business Intelligence \& Data Analytics, Module 11 Final Project

> Educational coursework demo. Not a medical device and not a substitute for clinical diagnosis.

\---

## Dataset

|||
|-|-|
|Source|Kaggle — Breast Cancer Analysis|
|Origin|University of Calabar Teaching Hospital Cancer Registry|
|Period|January 2019 – August 2021|
|Records|213 raw → 211 after cleaning|
|Features|11 raw → 8 predictors|
|Target|`Diagnosis Result` — Benign (119) / Malignant (92)|

### Predictors

|Feature|Type|Description|
|-|-|-|
|Age|Numeric|Patient age at diagnosis (13–77)|
|Menopause|Binary|Menopausal status at diagnosis|
|Tumor Size (cm)|Numeric|Size of the excised tumour|
|Inv-Nodes|Numeric|Axillary lymph nodes containing metastatic tissue|
|Breast|Categorical|Left / Right|
|Metastasis|Binary|Cancer present elsewhere in the body|
|Breast Quadrant|Categorical|Upper/Lower × Inner/Outer|
|History|Binary|Family history of breast cancer|

`S/N` and `Year` were dropped as identifiers carrying no signal.

\---

## Methodology

### Preprocessing

The raw file used `#` as a placeholder for missing data across both numeric and categorical columns. Handling:

* Categorical columns — stripped, title-cased, `#` mapped to an explicit `Unknown` level
* Numeric columns — `#` coerced to `NaN` via `pd.to\_numeric(errors='coerce')`
* Remaining rows with nulls dropped (213 → 211)

All transformations are wrapped in a `ColumnTransformer` inside a `Pipeline`, so imputation, scaling, and one-hot encoding are fitted **only on training folds**. This prevents data leakage during cross-validation.

```
ColumnTransformer
├── num → SimpleImputer(median) → StandardScaler
└── cat → SimpleImputer(most\_frequent) → OneHotEncoder(handle\_unknown='ignore')
```

### Train/test split

Stratified 80/20 split — 168 train / 43 test. Class balance held at 56.5/43.5 in train vs 55.8/44.2 in test.

### Models compared

Logistic Regression · Decision Tree · Random Forest · SVM · KNN

Each was tuned with `GridSearchCV` (5-fold stratified, F1 scoring), then validated with `RepeatedStratifiedKFold` (5 splits × 10 repeats) for a stable estimate.

\---

## Results

### Repeated cross-validation (50 folds, F1)

|Model|Mean ± Std|Range|
|-|-|-|
|Decision Tree|0.877 ± 0.062|0.720 – 0.966|
|SVM|0.876 ± 0.065|0.750 – 1.000|
|KNN|0.875 ± 0.064|0.750 – 1.000|
|Logistic Regression|0.873 ± 0.065|0.750 – 1.000|
|Random Forest|0.862 ± 0.064|0.741 – 0.966|

All five models are statistically indistinguishable. The 0.015 spread sits well inside a standard deviation of \~0.064.

### Nested cross-validation

**0.840 ± 0.036** — the honest generalization estimate.

The \~0.04 gap below the standard CV figure of 0.877 is optimism introduced by hyperparameter tuning; the grid search partially fitted the validation folds. Nested CV removes that bias by keeping model selection inside an inner loop.

### Held-out test set (Decision Tree, n=43)

|Metric|Value|
|-|-|
|Accuracy|0.907|
|Precision (Malignant)|0.941|
|Recall (Malignant)|0.842|
|F1 (Malignant)|0.889|
|ROC-AUC|0.958|

Confusion matrix: **TN=23, FP=1, FN=3, TP=16**

The three false negatives are the errors that matter. In a diagnostic context a missed malignancy costs far more than a false alarm sent for follow-up imaging.

\---

## Key findings

**Threshold tuning matters more than model choice.** The tuned Decision Tree has only four leaves, producing just four distinct probabilities (0.00, 0.19, 0.64, 1.00). Any threshold between 0.20 and 0.60 gives identical predictions — the cutoff is effectively uncontrollable. Logistic Regression yields continuous probabilities and makes the trade-off visible:

|Threshold|Recall|Precision|Missed|False alarms|
|-|-|-|-|-|
|0.3|0.842|0.941|3|1|
|0.5|0.789|1.000|4|0|

Moving from 0.5 to 0.3 catches one more cancer at the cost of one unnecessary follow-up. For a diagnostic screen that is the correct direction. The 0.5 default is a convention, not an optimum.

**Feature selection gave no improvement.** RFECV retained 9 of 9 encoded features and scored 0.858 ± 0.062 against 0.877 ± 0.062 for the full set — equivalent within noise. The pipeline places RFECV *inside* the cross-validation loop, so selection is refit per fold rather than on the full dataset.

**Regularization helps modestly and the type doesn't matter.**

|Configuration|F1|
|-|-|
|No penalty|0.841 ± 0.063|
|L2 (Ridge)|0.859 ± 0.061|
|L1 (Lasso)|0.860 ± 0.061|
|ElasticNet|0.859 ± 0.060|

Tuned best: `C=0.1, penalty='l1'` at F1 0.868. The regularization path shows classic overfitting — as `C` increases, train F1 climbs to 0.905 while validation drops to 0.829. L1 at `C=0.1` zeroed 10 of 14 coefficients, leaving Inv-Nodes, Tumor Size, Metastasis, and Age.

**Permutation importance** ranks Tumor Size (0.127) and Inv-Nodes (0.126) far above everything else, followed by Menopause (0.069) and Breast Quadrant (0.048). Breast side, Metastasis, and History contributed nothing measurable to the tree.

**Learning curve** shows train and validation converging with validation still climbing at 135 training examples — more data would improve performance.

\---

## Deployment

Gradio interface with three tabs:

* **Single patient** — sliders and dropdowns for each predictor, colour-coded result with a probability bar and certainty label
* **Batch (CSV)** — upload a file, get scored rows plus a downloadable `predictions.csv`
* **About** — dataset, method, and limitations

The model artifact is a dictionary containing the fitted pipeline, decision threshold, and feature list, serialized with `joblib`. Bundling the feature names guards against column-order drift at inference time.

```python
artifact = {
    'model': best\_model,       # fitted Pipeline
    'threshold': best\_threshold,
    'features': X\_train.columns.tolist(),
}
```

### Running locally

```bash
pip install -r requirements.txt
python app.py
```

### Files

```
├── app.py                        # Gradio interface
├── breast\_cancer\_model.pkl       # serialized pipeline + threshold + features
├── requirements.txt
└── M11\_FinalProject\_Breast\_Cancer.ipynb
```

`requirements.txt` pins `scikit-learn==1.6.1` to match the training environment — unpickling across versions fails.

\---

## Tech stack

**Python** · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Gradio

Key scikit-learn components: `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomizedSearchCV`, `RepeatedStratifiedKFold`, `RFECV`, `permutation\_importance`, `learning\_curve`

\---

## Limitations

* **Small sample.** 211 records, 43 in the test set. One patient shifts test accuracy by 2.3 points, so model rankings from the test set are not meaningful.
* **Single source.** One hospital registry over 24 months. No external validation cohort.
* **No calibration.** Predicted probabilities have not been calibrated, so they should be read as scores rather than true risk estimates.
* **Not clinically validated.** This is a coursework demonstration of an end-to-end ML workflow, not a diagnostic tool.

