# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:19:17 2025

@author: Felix
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import matplotlib.colors as mcolors

# ---------------- USER CONFIG ---------------- #
target = "binary_recovery"

output_root = r"E:\Thesis\data\MODEL\rf_results_regression\chosen"
train_file = r"E:\Thesis\data\shrink_metrics\chosen_pre8020\shrinkmetrics_v10_80_chose.gpkg"
vali_file = r"E:\Thesis\data\shrink_metrics\chosen_pre8020\shrinkmetrics_v10 _20_chose.gpkg"

exclude_features = ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
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
    'binary_recovery', 'geometry']

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

# Output folder
out_dir = os.path.join(output_root, target)
os.makedirs(out_dir, exist_ok=True)

# Load best parameters
param_path = os.path.join(out_dir, f"{target}_best_params.json")
with open(param_path, "r") as f:
    best_params = json.load(f)
print(f"✅ Loaded best parameters: {best_params}")

# Prepare data
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]
X_vali = df_vali.drop(columns=[target] + exclude_features, errors="ignore")
y_vali = df_vali[target]

# Store metrics
all_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
all_preds = []
all_probs = []
all_models = []

for i in range(n_runs):
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state_seed + i,
        n_jobs=-1,
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

# Save mean ± std for metrics
df_summary = pd.DataFrame({
    "Metric": list(all_metrics.keys()),
    "Mean": [np.mean(all_metrics[m]) for m in all_metrics],
    "Std": [np.std(all_metrics[m]) for m in all_metrics]
})
df_summary.to_csv(os.path.join(out_dir, f"{target}_rf_metrics.csv"), index=False)
print(df_summary)

# Select best model by F1
best_idx = np.argmax(all_metrics["F1"])
best_model = all_models[best_idx]
best_preds = all_preds[best_idx]
best_probs = all_probs[best_idx]

# Save classification results
results_df = pd.DataFrame({
    "plot_id": gdf_vali["plot_id"] if "plot_id" in gdf_vali.columns else np.arange(len(y_vali)),
    "true_label": y_vali.values,
    "predicted_label": best_preds,
    "prob_class_1": best_probs,
    "misclassified": (y_vali.values != best_preds).astype(int)
})
results_df.to_csv(os.path.join(out_dir, f"{target}_classification_results.csv"), index=False)

# Feature importance
importances = best_model.feature_importances_
feature_names = X_train.columns
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(top_n_features)
fi_df.to_csv(os.path.join(out_dir, f"{target}_feature_importance.csv"), index=False)

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
plt.xlabel("Importance")
plt.title(f"Top {top_n_features} Features for {target}")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target}_feature_importance.png"), dpi=300)
plt.close()

# Misclassification histogram
plt.figure(figsize=(8, 5))
plt.hist(results_df["misclassified"], bins=[-0.5, 0.5, 1.5], rwidth=0.7, color='orange', edgecolor='black')
plt.xticks([0, 1], ["Correct", "Misclassified"])
plt.title("Classification Errors")
plt.ylabel("Plot Count")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target}_error_distribution.png"), dpi=300)
plt.close()

# SHAP summary plot
print(f"📊 Generating SHAP summary for {target}...")
explainer = shap.Explainer(best_model, X_vali)
shap_values = explainer(X_vali, check_additivity=False)

# Safe plotting — SHAP will figure out the shape internally
shap.summary_plot(shap_values, X_vali, show=False, max_display=top_n_features)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target}_shap_summary.png"), dpi=300)
plt.close()


# ROC curve
fpr, tpr, thresholds = roc_curve(y_vali, best_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_vali, best_probs):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {target}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target}_roc_curve.png"), dpi=300)
plt.close()

# Spatial error map
gdf_vali_geom = gpd.read_file(vali_file)[["plot_id", "geometry"]]
results_geo = gdf_vali_geom.merge(results_df, on="plot_id")
gpkg_path = os.path.join(out_dir, f"{target}_error_map.gpkg")
results_geo.to_file(gpkg_path, driver="GPKG")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
results_geo.plot(
    column="misclassified",
    cmap=mcolors.ListedColormap(["lightgreen", "red"]),
    edgecolor="black",
    linewidth=0.2,
    legend=True,
    ax=ax,
    legend_kwds={'labels': ["Correct", "Misclassified"]}
)
ax.set_title(f"Spatial Classification Error Map for {target}")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"{target}_error_map.png"), dpi=300)
plt.close()
print(f"🗺️ Spatial classification error map saved.")

# %%
df_train['binary_recovery'].value_counts(normalize=True)

