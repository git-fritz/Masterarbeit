# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 22:15:27 2025

@author: Felix
"""

# -*- coding: utf-8 -*-
"""
XGBoost Classification with Threshold Tuning (Youden's J vs G-Mean)
Created on Mar 24, 2025 by Felix
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------- USER CONFIG ---------------- #
target = "binary_recovery"
target_folder = f"{target}_100"

output_root = r"E:\Thesis\data\MODEL\xgb_class_new_results\auto"
train_file = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_80.gpkg"
vali_file = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_20.gpkg"

exclude_features = [
    'id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
    'shrink_dis', 'PartID', 'area', 'centroid_x', 'centroid_y',
    'edge_points', 'side', 'SegmentID', 'plot_id', 'segment_area',
    'side_area', 'plot_area', 'plot_short_veg_%cover',
    'plot_medium_veg_%cover', 'plot_tall_veg_%cover', 'plot_forest_%cover',
    'side_short_veg_%cover', 'side_medium_veg_%cover',
    'side_tall_veg_%cover', 'side_forest_%cover',
    'segment_short_veg_%cover', 'segment_medium_veg_%cover',
    'segment_tall_veg_%cover', 'segment_forest_%cover', 'adjacency_area_ha',
    'adjacency_tree13m_coverage', 'adjacency_tree8m_coverage',
    'adjacency_tree5m_coverage', 'adjacency_tree3m_coverage',
    'adjacency_tree1.5m_coverage', 'hummock_coverage', 'hollow_coverage',
    'ndtm_iqr','mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
    'mean_chm_under2', 'median_chm_under2', 'tree_count', 'tree_density', 'trees_per_ha',
    'plot_pixels', 'veg_pixels_above_60cm', 'veg_cover_percent_above_60cm',
    'binary_recovery', 'geometry'
]  # (keep your full list here)

n_runs = 100
top_n_features = 10
random_state_seed = 42
# ------------------------------------------------ #

# Load data
gdf_train = gpd.read_file(train_file)
gdf_vali = gpd.read_file(vali_file)
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
gdf_vali = gdf_vali.drop(columns=["geometry"], errors="ignore")
df_train = gdf_train.select_dtypes(include=[np.number])
df_vali = gdf_vali.select_dtypes(include=[np.number])

# Output
out_dir = os.path.join(output_root, target_folder)
os.makedirs(out_dir, exist_ok=True)

# Load best params
param_path = os.path.join(out_dir, f"{target_folder}_best_params.json")
with open(param_path, "r") as f:
    best_params = json.load(f)
print(f"✅ Loaded best parameters: {best_params}")

# Prepare data
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]
X_vali = df_vali.drop(columns=[target] + exclude_features, errors="ignore")
y_vali = df_vali[target]

# Train 100 models
all_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
all_preds, all_probs, all_models = [], [], []

for i in range(n_runs):
    model = XGBClassifier(
        random_state=random_state_seed + i,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
        **best_params
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_vali)[:, 1]
    y_pred = model.predict(X_vali)

    all_metrics["Accuracy"].append(accuracy_score(y_vali, y_pred))
    all_metrics["Precision"].append(precision_score(y_vali, y_pred, zero_division=0))
    all_metrics["Recall"].append(recall_score(y_vali, y_pred))
    all_metrics["F1"].append(f1_score(y_vali, y_pred))
    all_metrics["AUC"].append(roc_auc_score(y_vali, y_prob))
    all_preds.append(y_pred)
    all_probs.append(y_prob)
    all_models.append(model)

# Save 100-model metrics summary
df_summary = pd.DataFrame({
    "Metric": list(all_metrics.keys()),
    "Mean": [np.mean(all_metrics[m]) for m in all_metrics],
    "Std": [np.std(all_metrics[m]) for m in all_metrics]
})
df_summary.to_csv(os.path.join(out_dir, f"{target_folder}_xgb_metrics.csv"), index=False)
print(df_summary)

# Select best model by F1
best_idx = np.argmax(all_metrics["F1"])
best_model = all_models[best_idx]
best_probs = all_probs[best_idx]

# ---------------- THRESHOLD TUNING ---------------- #
fpr, tpr, thresholds = roc_curve(y_vali, best_probs)
youden_j = tpr - fpr
gmean = np.sqrt(tpr * (1 - fpr))

idx_j = np.argmax(youden_j)
idx_g = np.argmax(gmean)

thr_j = thresholds[idx_j]
thr_g = thresholds[idx_g]

# Apply both thresholds
preds_j = (best_probs >= thr_j).astype(int)
preds_g = (best_probs >= thr_g).astype(int)

f1_j = f1_score(y_vali, preds_j)
f1_g = f1_score(y_vali, preds_g)

print(f"\nYouden J: threshold={thr_j:.3f}, F1={f1_j:.3f}")
print(f"G-Mean  : threshold={thr_g:.3f}, F1={f1_g:.3f}")

# Decide best threshold
if f1_g > f1_j:
    final_preds = preds_g
    final_threshold = thr_g
    best_method = "G-Mean"
else:
    final_preds = preds_j
    final_threshold = thr_j
    best_method = "Youden's J"

# Save final threshold + method
with open(os.path.join(out_dir, f"{target_folder}_optimal_threshold.json"), "w") as f:
    json.dump({"optimal_threshold": float(final_threshold), "method": best_method}, f)

# Save threshold tuning comparison
threshold_df = pd.DataFrame({
    "threshold": thresholds,
    "youden_j": youden_j,
    "gmean": gmean
})
threshold_df.to_csv(os.path.join(out_dir, f"{target_folder}_threshold_comparison.csv"), index=False)

# Plot threshold comparison
plt.figure(figsize=(10, 6))
plt.plot(thresholds, youden_j, label="Youden's J", color='blue')
plt.plot(thresholds, gmean, label="G-Mean", color='green')
plt.axvline(thr_j, linestyle='--', color='blue', alpha=0.5)
plt.axvline(thr_g, linestyle='--', color='green', alpha=0.5)
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization: Youden's J vs G-Mean")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_threshold_comparison.png"), dpi=300)
plt.close()

# Save final classification results
results_df = pd.DataFrame({
    "plot_id": gdf_vali["plot_id"] if "plot_id" in gdf_vali.columns else np.arange(len(y_vali)),
    "true_label": y_vali.values,
    "prob_class_1": best_probs,
    "thresholded_label": final_preds,
    "thresholded_misclassified": (y_vali.values != final_preds).astype(int)
})
results_df.to_csv(os.path.join(out_dir, f"{target_folder}_thresholded_results.csv"), index=False)

# Save classification report as CSV
report_dict = classification_report(y_vali, final_preds, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(out_dir, f"{target_folder}_classification_report.csv"))

# Feature importance
importances = best_model.feature_importances_
feature_names = X_train.columns
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(top_n_features)
fi_df.to_csv(os.path.join(out_dir, f"{target_folder}_feature_importance.csv"), index=False)

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
plt.xlabel("Importance")
plt.title(f"Top {top_n_features} Features for {target}")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_feature_importance.png"), dpi=300)
plt.close()

# Misclassification histogram
plt.figure(figsize=(8, 5))
plt.hist(results_df["thresholded_misclassified"], bins=[-0.5, 0.5, 1.5], rwidth=0.7, color='orange', edgecolor='black')
plt.xticks([0, 1], ["Correct", "Misclassified"])
plt.title("Classification Errors (Thresholded)")
plt.ylabel("Plot Count")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_error_distribution.png"), dpi=300)
plt.close()

# SHAP plot
print(f"📊 Generating SHAP summary for {target}...")
explainer = shap.Explainer(best_model, X_vali)
shap_values = explainer(X_vali, check_additivity=False)
shap.summary_plot(shap_values, X_vali, show=False, max_display=top_n_features)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_shap_summary.png"), dpi=300)
plt.close()

# ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_vali, best_probs):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {target}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_roc_curve.png"), dpi=300)
plt.close()

# Spatial map
gdf_vali_geom = gpd.read_file(vali_file)[["plot_id", "geometry"]]
results_geo = gdf_vali_geom.merge(results_df[["plot_id", "thresholded_misclassified"]], on="plot_id")
gpkg_path = os.path.join(out_dir, f"{target_folder}_error_map.gpkg")
results_geo.to_file(gpkg_path, driver="GPKG")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
results_geo.plot(
    column="thresholded_misclassified",
    cmap=mcolors.ListedColormap(["lightgreen", "red"]),
    edgecolor="black",
    linewidth=0.2,
    legend=False,
    ax=ax
)
legend_elements = [
    mpatches.Patch(color="lightgreen", label="Correct"),
    mpatches.Patch(color="red", label="Misclassified")
]
ax.legend(handles=legend_elements, loc="lower left")
ax.set_title(f"Spatial Classification Error Map for {target}")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target_folder}_error_map.png"), dpi=300)
plt.close()

print("✅ All done.")
