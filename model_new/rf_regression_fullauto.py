# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 22:44:13 2025

@author: Felix
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.colors as mcolors

# ---------------- USER CONFIG ---------------- #
target_variables = ["median_chm","mean_chm", "tree_count"]
# data_folder = r"E:\Thesis\code_storage\Masterarbeit\model_results\auto"
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
       'binary_recovery', 'geometry']  # <-- your DEM_only exclusion list

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

for target in target_variables:
    print(f"\n🔁 Running RF for target: {target}")

    # Output folder
    out_dir = os.path.join(output_root, target)
    os.makedirs(out_dir, exist_ok=True)

    # Load best parameters from JSON
    param_path = os.path.join(out_dir, f"{target}_best_params.json")
    with open(param_path, "r") as f:
        best_params = json.load(f)
    print(f"✅ Loaded best parameters for {target}: {best_params}")

    # Prepare data
    X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
    y_train = df_train[target]
    X_vali = df_vali.drop(columns=[target] + exclude_features, errors="ignore")
    y_vali = df_vali[target]

    all_metrics = {"MAE": [], "RMSE": [], "R2": []}
    all_preds = []
    all_models = []

    for i in range(n_runs):
        model = RandomForestRegressor(
            random_state=random_state_seed + i,
            n_jobs=-1,
            **best_params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_vali)

        all_metrics["MAE"].append(mean_absolute_error(y_vali, y_pred))
        all_metrics["RMSE"].append(np.sqrt(mean_squared_error(y_vali, y_pred)))
        all_metrics["R2"].append(r2_score(y_vali, y_pred))
        all_preds.append(y_pred)
        all_models.append(model)

    # Save metrics summary
    df_summary = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Mean": [np.mean(all_metrics["MAE"]), np.mean(all_metrics["RMSE"]), np.mean(all_metrics["R2"])],
        "Std": [np.std(all_metrics["MAE"]), np.std(all_metrics["RMSE"]), np.std(all_metrics["R2"])]
    })
    df_summary.to_csv(os.path.join(out_dir, f"{target}_rf_metrics.csv"), index=False)
    print(df_summary)

    # Select best model
    best_idx = np.argmax(all_metrics["R2"])
    best_model = all_models[best_idx]
    best_preds = all_preds[best_idx]
    residuals = y_vali - best_preds

    # Save residuals
    residuals_df = pd.DataFrame({
        "plot_id": gdf_vali["plot_id"] if "plot_id" in gdf_vali.columns else np.arange(len(y_vali)),
        f"{target}_actual": y_vali.values,
        f"{target}_predicted": best_preds,
        "residual": residuals
    })
    residuals_df.to_csv(os.path.join(out_dir, f"{target}_residuals.csv"), index=False)

    # Feature importance
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n_features)
    fi_df.to_csv(os.path.join(out_dir, f"{target}_feature_importance.csv"), index=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n_features} Features for {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_feature_importance.png"), dpi=300)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='steelblue', edgecolor='black')
    plt.title(f"Residual Distribution for {target}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_residual_distribution.png"), dpi=300)
    plt.close()

    # Residuals vs. predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(best_preds, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs. Predicted for {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_residuals_vs_predicted.png"), dpi=300)
    plt.close()

    # SHAP summary plot
    print(f"📊 Generating SHAP summary for {target}...")
    explainer = shap.Explainer(best_model, X_vali)
    shap_values = explainer(X_vali, check_additivity=False)
    shap.summary_plot(shap_values, X_vali, show=False, max_display=top_n_features)
    plt.tight_layout()
    shap_path = os.path.join(out_dir, f"{target}_shap_summary.png")
    plt.savefig(shap_path, dpi=300)
    plt.close()

    # Spatial residuals map
    gdf_vali_geom = gpd.read_file(vali_file)
    gdf_vali_geom = gdf_vali_geom[["plot_id", "geometry"]]  # minimal join keys

    residuals_geo = gdf_vali_geom.merge(residuals_df, on="plot_id")
    gpkg_path = os.path.join(out_dir, f"{target}_residuals_map.gpkg")
    residuals_geo.to_file(gpkg_path, driver="GPKG")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    divider = max(abs(residuals_geo["residual"].min()), residuals_geo["residual"].max())
    residuals_geo.plot(
        column="residual",
        cmap="RdBu",
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        ax=ax,
        legend_kwds={'label': "Residual (Actual - Predicted)", 'shrink': 0.6},
        vmin=-divider,
        vmax=divider
    )
    ax.set_title(f"Spatial Residuals Map for {target}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target}_residuals_map.png"), dpi=300)
    plt.close()
    print(f"🗺️ Residuals spatial map saved.")
