# Customer Review Sentiment Analysis — IMDb Movie Reviews

**Assignment 10 — M1\_A10**
Notebook: `M1\_A10\_Customer\_Review\_Sentiment\_Analysis\_IMDB\_movie.ipynb` (Google Colab)

An end-to-end NLP pipeline on IMDb movie reviews: text preprocessing, three vectorization
techniques, traditional classifiers with hyperparameter tuning, and a fine-tuned BERT model.

\---

## Dataset

|Item|Value|
|-|-|
|Source|IMDb Reviews sentiment dataset (`IMDB Dataset.csv`)|
|Rows loaded|2,000|
|Original columns|`review`, `sentiment`|
|Duplicates found|0|
|Missing values|0|

## Environment

Python 3.13 (Colab), GPU runtime (T4) for the BERT fine-tuning step.

```
pandas · numpy · matplotlib · seaborn
nltk · spacy · textblob · contractions
scikit-learn · gensim 4.4.0
tensorflow 2.20.0 · torch · transformers · datasets · accelerate
```

NLTK downloads required: `punkt`, `punkt\_tab`, `averaged\_perceptron\_tagger`,
`averaged\_perceptron\_tagger\_eng`, `wordnet`, `omw-1.4`, `stopwords`.

\---

## Part 1 — Dataset Selection and Preprocessing

A working copy (`imdbreviews\_clone`) was created so the raw data stays intact.

Preprocessing pipeline applied to each review:

1. Remove HTML tags (`<br />`, etc.)
2. Expand contractions
3. Lowercase
4. Strip punctuation and digits
5. Tokenize (`word\_tokenize`)
6. Remove English stopwords
7. POS-tag and lemmatize (WordNet lemmatizer with Treebank→WordNet POS mapping)

Porter stemming was also applied for comparison (`stemmed\_tokens`), and the lemmatized
tokens were joined back into `text\_lemmatized` for the vectorizers.

**Label generation.** Sentiment labels were derived with TextBlob polarity scores:

|Category|Rule|Count|
|-|-|-|
|Favorable|polarity > 0|1,497|
|Unfavorable|polarity < 0|502|
|No Impact|polarity == 0|1|

\---

## Part 2 — Feature Engineering

|Technique|Tool|Output shape|
|-|-|-|
|Bag of Words|`CountVectorizer()`|(2000, 25136) — 99.46% zeros|
|TF-IDF|`TfidfVectorizer(max\_features=5000, stop\_words='english')`|(2000, 5000)|
|Word embeddings|`gensim.Word2Vec(vector\_size=100, window=5, min\_count=1)`|vocab 24,074|
|Document vectors|Mean of word vectors per review|(2000, 100)|

**Visualization.** The 200 most frequent words were projected to 2D with both PCA
(PC1 72.7%, PC2 24.6% of variance) and t-SNE (`perplexity=30`, `init='pca'`).

PCA is a linear projection, so its axes carry interpretable variance percentages.
t-SNE is nonlinear and preserves local neighbourhoods — clusters look tighter, but
the axes are arbitrary and between-cluster distances carry no meaning.

\---

## Part 3 — Text Classification Using Traditional Models

Features: Bag of Words (25,136 columns). Split: 80/20, `random\_state=5`.

### Baseline

|Model|Accuracy|Favorable F1|Unfavorable F1|
|-|-|-|-|
|Logistic Regression|0.8425|0.90|0.66|
|Linear SVM|0.8350|0.89|0.66|

### Grid search (5-fold CV, `scoring='f1\_macro'`)

|Model|Best parameters|Best CV score|
|-|-|-|
|Logistic Regression|`C=0.1, class\_weight='balanced', solver='liblinear'`|0.7921|
|Linear SVM|`C=0.01, class\_weight='balanced'`|0.7904|

### Tuned results

|Model|Accuracy|ROC-AUC|
|-|-|-|
|Logistic Regression|0.8350|0.8867|
|Linear SVM|0.8375|0.8901|

Tuning did not improve accuracy, which is common for linear models on high-dimensional
sparse text — the untuned defaults were already close to optimal.

\---

## Part 4 — Sentiment Analysis Using a Large Language Model

### Pre-trained inference (200-review sample)

|Model|Agreement with TextBlob labels|
|-|-|
|`distilbert-base-uncased-finetuned-sst-2-english`|0.585|
|`textattack/bert-base-uncased-SST-2`|0.580|

Both transformers predicted Unfavorable far more often than TextBlob did. Since these
models were trained on human-annotated SST-2 data and handle negation and sarcasm,
the disagreement most likely reflects weakness in the TextBlob labels rather than
error by the transformers.

### Fine-tuned BERT

`bert-base-uncased`, 2-class head, 1,599 train / 400 test, `max\_length=256`,
`learning\_rate=2e-5`, `batch\_size=16`, 3 epochs, `weight\_decay=0.01`, 100 warmup steps.

|Epoch|Training loss|Validation loss|Accuracy|F1 (weighted)|
|-|-|-|-|-|
|1|0.5032|0.3976|0.8100|0.7967|
|2|0.3021|0.4104|0.8175|0.8260|
|3|0.1600|0.4528|0.8250|0.8315|

Training runtime: 222 seconds on GPU.

Validation loss rose after epoch 1 while training loss kept falling — the model began
overfitting. Accuracy still improved slightly, but more epochs would not help.

\---

## Part 5 — Model Evaluation

Metrics computed: accuracy, macro precision/recall/F1, ROC-AUC, confusion matrices,
and ROC curves for both traditional classifiers.

|Model|Accuracy|Macro P|Macro R|Macro F1|ROC-AUC|
|-|-|-|-|-|-|
|Logistic Regression|0.8350|0.5221|0.5166|0.5191|0.8867|
|Linear SVM|0.8375|0.5246|0.5177|0.5209|0.8901|

The macro scores near 0.52 are an artifact, not a real result — see Limitations below.
ROC-AUC near 0.89 is the more meaningful number: both models separate the two real
classes well, and the gap between AUC and Unfavorable recall (\~0.65) suggests the
default 0.5 decision threshold is not optimal for the minority class.

\---

## Limitations

These are known issues in the current version, documented rather than hidden.

**1. Circular labels.** `sentiment\_category` was produced by TextBlob reading the same
review text that the features are built from. The classifiers are therefore learning to
reproduce TextBlob's lexicon rules, not to detect sentiment independently. The original
human-assigned `sentiment` column (positive/negative) was dropped during cloning and
would be the stronger ground truth.

**2. The "No Impact" class.** Only 1 review out of 2,000 received a polarity of exactly
zero. It remained in the train/test split for Part 3, so every macro-averaged metric
includes a class with a single sample scoring 0.00 — this is what pulls macro F1 down
from roughly 0.78 to 0.52. It was correctly excluded before the BERT fine-tuning step.

**3. Word2Vec similarity scores.** All nearest-neighbour similarities came back above
0.99 (`good → really, make, think`), which indicates a degenerate embedding space. With
only 2,000 documents and `min\_count=1`, there is not enough data to separate word
vectors meaningfully. A larger corpus or `min\_count=5+` would be needed.

**4. LLM sample size.** The transformer comparisons used 200 of 2,000 reviews to keep
CPU inference time manageable.

\---

## Reproducing

Run cells top to bottom. If variables appear stale, use **Runtime → Restart and run all** —
out-of-order execution is the most common source of errors in this notebook.

For the BERT section, set **Runtime → Change runtime type → T4 GPU** first. On CPU the
fine-tuning step takes over an hour instead of under four minutes.

