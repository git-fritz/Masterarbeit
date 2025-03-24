# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:22:08 2025

@author: Felix
"""
#regression

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# --- Config ---
target = "tree_count"
train_file = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_80.gpkg"
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

output_path = r"E:\Thesis\data\MODEL\xgb_results\auto\tree_count_100_best_params.json"

# --- Load and prep ---
gdf_train = gpd.read_file(train_file)
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
df_train = gdf_train.select_dtypes(include=[np.number])
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]

# --- Define XGBoost parameter grid ---
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# --- Run grid search ---
print("🔍 Starting XGBoost hyperparameter tuning...")
grid_search = RandomizedSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=42),
    param_distributions=param_grid,
    n_iter=100,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
grid_search.fit(X_train, y_train)
print("✅ Grid search complete.")
print("Best params:", grid_search.best_params_)

# --- Save best params ---
with open(output_path, "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)
print(f"Best parameters saved to {output_path}")
# %%
# classification
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# --- Config ---
target = "binary_recovery"
train_file = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_80.gpkg"
output_path = r"E:\Thesis\data\MODEL\xgb_results\auto\binary_recovery_100_best_params.json"

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

# --- Load and prep ---
gdf_train = gpd.read_file(train_file)
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
df_train = gdf_train.select_dtypes(include=[np.number])
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]

# --- Define XGBoost hyperparameter grid ---
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# --- Run grid search ---
print("🔍 Starting XGBoost hyperparameter tuning...")
grid_search = RandomizedSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='roc_auc',  # Or 'roc_auc' depending on your objective
    n_jobs=-1,
    verbose=1,
    random_state=42
)
grid_search.fit(X_train, y_train)

# --- Results ---
print("✅ Grid search complete.")
print("Best params:", grid_search.best_params_)

# --- Save best params ---
with open(output_path, "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)
print(f"Best parameters saved to {output_path}")

