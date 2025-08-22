# Comprehensive ML Algorithm Comparator

An interactive Gradio app (inside a Jupyter Notebook) to **compare ML algorithms** for **classification**, **regression**, and **clustering** on your own CSV data or quickly generated sample datasets. It computes rich metrics, plots side‑by‑side performance, surfaces feature importances for tree-based models, and suggests the best algorithm for your task.

> Notebook: `Comprehensive_ML_Algorithm_Comparator.ipynb`



## Key Features

- **Three task types:** classification, regression, clustering.
- **Plug in data quickly:** upload a CSV or generate synthetic data with adjustable samples/features.
- **Pick your models:** choose multiple algorithms per task and compare them in parallel.
- **Rich metrics:** cross‑validated scores plus task‑specific metrics (see below).
- **Visuals included:** comparison bar chart and feature‑importance plot (for tree-based models).
- **Best model hint:** highlights the top algorithm based on the primary metric for the task.
- **Caching & speed:** basic results caching and threaded evaluation for snappier comparisons.
- **XGBoost support:** Includes optional XGBoost models for classification and regression (auto-detected if `xgboost` is installed).



## Supported Algorithms

### Classification
ada_boost, decision_tree, extra_trees, gaussian_process, gradient_boosting, knn, lda, linear_svm, logistic_regression, mlp, naive_bayes, qda, random_forest, sgd, svm, xgboost

### Regression
ada_boost, decision_tree, elastic_net, extra_trees, gaussian_process, gradient_boosting, knn, lasso, linear_regression, linear_svr, mlp, random_forest, ridge, sgd, svm, xgboost

### Clustering
agglomerative, birch, dbscan, kmeans, spectral

> *Note:* Some algorithm parameter availability may vary by scikit‑learn version.



## Metrics

### Classification
- Accuracy, Balanced Accuracy
- Precision (macro/weighted), Recall (macro/weighted), F1 (macro/weighted)
- ROC AUC (binary or OvR/OvO for multiclass if `predict_proba` available)
- Log Loss (if `predict_proba` available)
- Matthews Correlation Coefficient, Cohen’s Kappa
- Cross‑val: Accuracy (mean/std), F1‑weighted (mean/std)

### Regression
- MSE, RMSE, MAE, MAPE
- R², Explained Variance, Median Absolute Error, Max Error
- *(If positive targets/predictions)* Mean Poisson Deviance, Mean Gamma Deviance
- Cross‑val: R² (mean/std), MAE (mean/std)
- Spearman correlation

### Clustering
- Silhouette, Calinski–Harabasz, Davies–Bouldin
- *(If true labels provided in data)* Adjusted Rand, Normalized Mutual Info, Homogeneity, Completeness, V‑measure

## Requirements

- Python 3.9+ (recommended)
- `gradio`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Optional: `xgboost` (for XGBoost models)

Install:
```bash
pip install gradio pandas numpy matplotlib scikit-learn
# optional
pip install xgboost
```


## ▶ How to Run

### Option A — In Jupyter
1. Open `Comprehensive_ML_Algorithm_Comparator.ipynb` in Jupyter or VS Code.
2. Run all cells. A Gradio UI will launch (or display inline if supported).

### Option B — As a script (quick start)
If you export the main cell content to a `.py` file with the `if __name__ == "__main__": demo.launch()`, run:
```bash
python comparator.py
```
Then open the printed local URL in your browser.


## Data Format (CSV Uploads)

- **For classification/regression:** last column = target `y`; all other columns = features `X`.
- **For clustering:** all columns are treated as features `X` (no target).

Tips:
- Ensure no mixed text/number columns unless you’ve encoded them beforehand.
- The app applies a standard scaler; label encoding is used if needed for targets.


## 🖱 Using the App

1. **Task Type:** choose *classification*, *regression*, or *clustering*.
2. **Data:** check **Use Sample Data** (set samples & features), or upload a CSV.
3. **Algorithms:** select one or more from the list (XGBoost appears if installed).
4. *(If XGBoost shown)* adjust learning rate, max depth, subsample, and colsample—updates apply live.
5. Click **Compare** to compute metrics, view the comparison chart, and see the **Best Algorithm** suggestion.
6. For tree-based models, inspect **Feature Importance** (top features per selected model).


## ⚙ Notes & Limitations

- **Parallelism:** uses a thread pool (up to 4 workers) to evaluate models in parallel.
- **Caching:** results cached per (task, data hash) to speed up repeated runs.
- **Cross‑validation:** 3‑fold CV for select metrics; adjust in code if you need more robust estimates.
- **Version quirks:** certain estimators/params (e.g., `n_jobs` or feature importance attributes) can differ between scikit‑learn releases.
- **Large data:** for very large CSVs, consider downsampling or increasing resources; training all models may be slow.
- **Probabilities:** metrics like ROC AUC/Log Loss require `predict_proba`; not all classifiers provide it.
