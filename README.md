# Comprehensive-ML-Algorithm-Comparator-with-Gradio
n interactive Gradio app (inside a Jupyter Notebook) to **compare ML algorithms** for **classification**, **regression**, and **clustering** on your own CSV data or quickly generated sample datasets. It computes rich metrics, plots side‑by‑side performance, surfaces feature importances for tree-based models, and suggests the best algorithm for your task.

> Notebook: `Comprehensive_ML_Algorithm_Comparator.ipynb`

---

## ✨ Key Features

- **Three task types:** classification, regression, clustering.
- **Plug in data quickly:** upload a CSV or generate synthetic data with adjustable samples/features.
- **Pick your models:** choose multiple algorithms per task and compare them in parallel.
- **Rich metrics:** cross‑validated scores plus task‑specific metrics (see below).
- **Visuals included:** comparison bar chart and feature‑importance plot (for tree-based models).
- **Best model hint:** highlights the top algorithm based on the primary metric for the task.
- **Caching & speed:** basic results caching and threaded evaluation for snappier comparisons.
- **XGBoost support:** {xgb_available_block}

---

## 🧠 Supported Algorithms

### Classification
### Regression
### Clustering


> *Note:* Some algorithm parameter availability may vary by scikit‑learn version.

---

## 📊 Metrics

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

---


# How to Run
Open Comprehensive_ML_Algorithm_Comparator.ipynb in Jupyter or VS Code.

Run all cells. A Gradio UI will launch (or display inline if supported).




## 📦 Requirements

- Python 3.9+ (recommended)
- `gradio`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Optional: `xgboost` (for XGBoost models)

Install:
```bash
pip install gradio pandas numpy matplotlib scikit-learn
# optional
pip install xgboost


