# Predicting Salary Bands in the AI Job Market

A supervised classification project that predicts whether an AI/data professional falls into a **Low, Medium or High** salary band, and tests whether daily use of AI tools actually pays more.

!\[Python](https://img.shields.io/badge/Python-3.11-blue)
!\[scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
!\[pandas](https://img.shields.io/badge/pandas-2.x-150458)
!\[License](https://img.shields.io/badge/License-MIT-green)

\---

## Business Problem

Compensation in the AI job market varies widely across roles like LLM Engineer, Machine Learning Engineer and Data Analyst, and candidates have little reliable benchmarking. A recurring claim in 2026 hiring discourse is that fluency with AI tooling commands a premium — or, conversely, that it prices workers out.

This project asks two questions:

1. **Can salary band be predicted** from role, seniority, industry and working-pattern attributes?
2. **Does daily AI tool use move the needle** on pay, once seniority is accounted for?

Salary is reframed from a continuous target into three bands because that is how recruiters and candidates actually benchmark — against ranges, not exact figures.

\---

## Dataset

|||
|-|-|
|**Source**|Google Dataset Search — *AI in Software Development Statistics 2026*|
|**File**|`ai\_ds\_job\_salaries\_2026.csv`|
|**Size**|1,000 rows × 27 columns|
|**Target**|`salary\_usd`, binned at the 33rd/67th percentiles|
|**Features used**|10 (5 categorical, 5 numeric)|
|**Missing values**|0|
|**Duplicates**|0|

**Salary band boundaries:** Low $12,000–$71,730 · Medium $71,730–$117,271 · High $117,271–$329,944
Classes are balanced by construction (334 / 332 / 334), giving a **baseline accuracy of 33.5%**.

\---

## Approach

**Preprocessing**

* One-hot encoding for categorical features, with `handle\_unknown='ignore'` for unseen job titles
* `StandardScaler` for the distance-based SVM; passthrough for tree models
* All transforms wrapped in a `ColumnTransformer` + `Pipeline` so nothing is fitted on test data
* Stratified 80/20 split (800 train / 200 test)

**Exploratory analysis**

* Distribution and outlier checks on the continuous target
* Pearson correlation for numeric features; **Cramér's V** for categorical association with the salary band
* Pivot table of mean salary across experience level × education level
* Collinearity check between the two AI-usage features

**Models**

|Model|Configuration|
|-|-|
|Decision Tree|`max\_depth=4`, `min\_samples\_leaf=5`|
|Random Forest|200 trees, `max\_features='sqrt'`|
|SVM|RBF kernel, `C=1.0`, `gamma='scale'`|

**Evaluation** — accuracy, per-class precision/recall/F1, confusion matrices, one-vs-rest ROC-AUC, 5-fold stratified cross-validation, and an explicit train-vs-test overfitting check.

\---

## Results

|Model|Test accuracy|5-fold CV|Train|Train–test gap|
|-|-|-|-|-|
|Decision Tree|0.495|**0.536 (± 0.023)**|0.570|**0.075**|
|Random Forest|**0.525**|0.520 (± 0.038)|1.000|0.475|
|SVM|0.510|0.528 (± 0.021)|0.689|0.179|

All three models beat the 33.5% baseline by 17–19 points. Critically, **the single split and cross-validation disagree on the ranking** — nothing about the models changed, only which rows landed in the test set. All three sit within one standard deviation of each other, so no model is genuinely superior.

The **pruned Decision Tree is the recommended model**: it leads on cross-validation, has by far the smallest train–test gap, and is directly explainable to a non-technical audience. Random Forest scored a perfect 1.000 on training data — textbook memorisation.

**Selected ROC-AUC (SVM, one-vs-rest):** Low 0.720 · Medium 0.575 · High 0.737

\---

## Key Findings

**1. Experience explains almost everything.**
`years\_experience` is the strongest numeric predictor (r = 0.447) and `experience\_level` the strongest categorical one (Cramér's V = 0.337). Mean salary climbs monotonically from $62,582 at Entry to $157,613 at Executive.

**2. AI tool use does not pay a premium.**
Daily AI tool users average $105,308 versus $91,071 for non-users — but the association with salary band is weak (V = 0.126), and all three AI-related variables land in weak-to-negligible territory. The raw gap is largely confounded with seniority: senior people use AI tools more. **AI fluency neither pays you more nor prices you out. Experience does the work.**

**3. Education matters far less than expected.**
`education\_level` scores V = 0.092 — effectively noise. Within any seniority row of the experience × education pivot, salary varies far less across education columns than it does down the seniority axis.

**4. The middle band is unlearnable with these features.**
Recall on the Medium class is 0.33–0.35 across all three models, with errors spreading evenly in both directions and AUC of just 0.575. Three different algorithm families failing identically is a **feature problem, not a model problem** — and it caps overall accuracy near 0.53.

**5. Errors respect the ordinal structure.**
True High predicted as Low occurred only 2–7 times per model. Confusion concentrates between neighbouring bands, meaning the models get the direction right even when they miss the exact band.

\---

## Repository Structure

```
.
├── data/
│   └── ai\_ds\_job\_salaries\_2026.csv
├── notebooks/
│   └── M11\_A8\_Supervised\_Learning\_Classification.ipynb
├── reports/
│   └── A8\_Supervised\_Learning\_Classification\_Report.md
├── requirements.txt
└── README.md
```

\---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `Google Colab`

\---

## Running It

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
jupyter notebook notebooks/M11\_A8\_Supervised\_Learning\_Classification.ipynb
```

Or open the notebook directly in Google Colab and upload `ai\_ds\_job\_salaries\_2026.csv` to the session.

\---

## Limitations

* **Sample size.** Analysis uses the first 1,000 rows; job title and industry have 12 and 10 levels respectively, leaving thin cells that inflate Cramér's V.
* **Accuracy ceiling.** \~53% is well short of production-grade. The dataset lacks the variables that most plausibly drive pay.
* **Fixed cut-points.** Band boundaries reflect 2026 percentiles and will drift as wages inflate.
* **Self-reported data.** Salary and satisfaction fields come from survey responses and carry the usual reporting bias.

\---

## Next Steps

* Add cost-of-living-adjusted location, company revenue/funding stage, and tenure at current employer — feature engineering is the highest-leverage improvement here, not further tuning
* Test an ordinal classifier or ordinal regression to exploit the natural Low < Medium < High ordering
* Deploy the pruned Decision Tree behind a FastAPI endpoint returning class probabilities, suppressing low-confidence predictions in favour of a group median
* Monitor population stability on incoming `years\_experience` and `job\_title`, and recompute band cut-points quarterly

\---

## Author

Ayleen Santander

Business Intelligence \& Data Analytics | Willis College Online
Assignment 8: Supervised Learning Classification

