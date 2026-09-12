# Customer Segmentation with Unsupervised Learning

Clustering and dimensionality reduction on a marketing campaign dataset (2,216 customers) to find natural customer segments.

**Author:** Ayleen Santander
**Course:** Assignment 9 — Unsupervised Learning

\---

## Dataset

`marketing\_campaign.csv` — tab-separated, 2,240 rows, 29 columns. Contains customer demographics, spending by product category, and purchase channel counts.

After dropping rows with missing `Income`: **2,216 rows**.

### Features used (9)

`Income`, `NumWebVisitsMonth`, `Recency`, `Year\_Birth`, `Kidhome`, `Teenhome`, `MntWines`, `MntMeatProducts`, `MntGoldProds`

A derived column `TotalPurchasesMonth` (sum of the four purchase-channel columns) was also created.

\---

## Methods

|Step|Technique|
|-|-|
|Preprocessing|`dropna()`, `StandardScaler`|
|Choosing k|Elbow method (inertia), Silhouette score|
|Clustering|K-Means (k=2, k=3), Agglomerative (Ward), DBSCAN|
|Dimensionality reduction|PCA (2D), t-SNE (2D)|
|Evaluation|Silhouette, Calinski-Harabasz, Davies-Bouldin|

\---

## Results

### Optimal k

* **Elbow method** → bend at k=3
* **Silhouette** → best at k=2 (0.267)

### Model comparison

|Model|Clusters|Noise|Silhouette|Calinski-Harabasz|Davies-Bouldin|
|-|-|-|-|-|-|
|KMeans (k=3)|3|0|0.234|687.20|1.617|
|**KMeans (k=2)**|2|0|**0.267**|**844.29**|**1.535**|
|Hierarchical (k=3)|3|0|0.201|612.56|1.771|
|DBSCAN (eps=0.5)|10|1910|0.048|30.49|0.967|

**KMeans with k=2 performs best** on Silhouette and Calinski-Harabasz, and has the lowest Davies-Bouldin among the non-noise models.

DBSCAN's low Davies-Bouldin is misleading — `eps=0.5` is too tight for 9 standardized dimensions, so it flagged 1,910 of 2,216 points as noise and scored only the compact remainder.

### PCA

* PC1 = 36.6% variance, PC2 = 15.9% → **52.5% in 2D**
* 5 components needed for 80% variance, 7 for 90%

Going from 9 features to 7 is barely a reduction. The data has no small set of dominant patterns — variance is spread thinly across many directions, which matches the modest silhouette scores.

\---

## Key takeaways

1. The segments are real but not sharply separated (silhouette \~0.27 is weak-to-moderate).
2. PCA is linear with interpretable axes and meaningful distances; t-SNE preserves only local neighbourhoods, so cluster sizes and gaps on a t-SNE plot carry no information.
3. k=2 wins on metrics, but k=3 may be more useful in practice — profile both on the original features and pick the one whose segments a business owner can actually name.

\---

## Deployment notes

Deployed as a segmentation API: new records are scaled with the **saved** `StandardScaler` and assigned a cluster label.

* **Latency** — scoring is fast; the surrounding pipeline (fetching, scaling) is the bottleneck.
* **Scalability** — scoring scales well; re-training does not. Use `MiniBatchKMeans` or sampling on large data.
* **Maintenance** — scaler, model, and column ordering must be versioned together. Refitting the scaler in production silently shifts every assignment, and a column-order mismatch returns wrong labels without erroring.
* **Cluster ID drift** — IDs reshuffle on every re-fit. Remap by centroid proximity or dashboards will swap segments silently.

### Monitoring

Clustering is unsupervised, so nothing alerts you when it breaks. Track cluster sizes, mean distance to centroid, and input drift against the training baseline. Re-fit quarterly or when a metric crosses threshold, validate the new model on silhouette and segment stability before promoting, and keep the previous version for rollback.

\---

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn yellowbrick
```

Open `M11.A9\_Unsupervised\_Learning.ipynb` in Google Colab, upload `marketing\_campaign.csv` to `/content/`, then **Runtime → Restart and run all**.

\---

## Libraries

`numpy` · `pandas` · `matplotlib` · `seaborn` · `scikit-learn` (KMeans, AgglomerativeClustering, DBSCAN, PCA, TSNE, StandardScaler, clustering metrics) · `yellowbrick`

