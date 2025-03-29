# ML Modeling Scripts – Regrowth Prediction

This folder contains scripts for running **Random Forest (RF)** and **XGBoost (XGB)** models to predict vegetation regrowth on seismic lines using LiDAR-derived microtopographic and vegetation features.

---

## 📁 Scripts Overview

| Script | Description |
|--------|-------------|
| `rf_hyperparameter.py` | Random Forest hyperparameter tuning (regression and classification). |
| `xgb_hyperparameter.py` | XGBoost hyperparameter tuning (regression and classification). |
| `rf_regression_fullauto.py` | Runs 100 RF regression models for `mean_chm`, `median_chm`, and `tree_count` (plot-based). |
| `rf_regression_fullauto_100.py` | Same as above but for segment-based predictions using `_100` logic. |
| `rf_classification_fullauto.py` | RF classification for binary regrowth (`binary_recovery`), basic version. |
| `rf_classification_fullauto_100.py` | Enhanced RF classification with threshold tuning (Youden's J, G-Mean). |
| `xgb_regression_fullauto.py` | XGB regression for all three targets (plot-based). |
| `xgb_regression_fullauto_100.py` | XGB regression for segment-based predictions. |
| `xgb_classification_fullauto.py` | XGB classification with ROC-based threshold optimization. |
| `xgb_classification_fullauto_100.py` | Improved version of above with standardized output naming and threshold tuning. |

---

## 🧠 Model Targets

### Regression
- `mean_chm` – Mean canopy height
- `median_chm` – Median canopy height
- `tree_count` – Number of trees per segment or plot

### Classification
- `binary_recovery` – Binary class for vegetation recovery (>60 cm)

---

## 🧪 Features

- 100 independent model runs per script for robust evaluation
- SHAP value explanation and feature importance ranking
- Residuals and diagnostics (for regression)
- Threshold tuning based on:
  - **Youden’s J**
  - **G-Mean**
- Auto/manual and plot/segment naming conventions
- Unified output logic across RF and XGB

---

## 📤 Output Structure

Each model writes to a clearly structured directory, such as:

```
/MODEL/{model_type}_results/{auto|chosen}/{target}_100/
├── {model_id}_metrics.csv
├── {model_id}_residuals.csv (regression)
├── {model_id}_thresholded_results.csv (classification)
├── {model_id}_feature_importance.csv
├── {model_id}_optimal_threshold.json
├── {model_id}_threshold_comparison.csv
├── *.png (plots)
```

---

## 📦 Dependencies

All scripts rely on:

- `numpy`
- `pandas`
- `geopandas`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`

Install them via:

```bash
pip install numpy pandas geopandas scikit-learn xgboost shap matplotlib
```

---

## 📌 Notes

- Input data: Pre-split `.gpkg` files (80/20 train/validation).
- Feature exclusion is tailored to LiDAR/DEM-based features only.
- Classification models include optional ROC-based threshold optimization.
- Plots are automatically saved as PNGs for publication-ready figures.

---

Feel free to adapt the configs in each script for other targets or input datasets.  
Authored by Felix | Thesis Project (2025)