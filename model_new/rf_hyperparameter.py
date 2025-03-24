# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:24:58 2025

@author: Felix
"""
# %%
# hyperparameter tuning for regression

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# --- Config ---
target = "tree_count"  # Change to mean_chm or tree_count as needed
train_file = r"E:\Thesis\data\shrink_metrics\chosen_pre8020\shrinkmetrics_v10_80_chose.gpkg"
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
       'binary_recovery', 'geometry']  # your DEM_only exclusion list

output_path = r"E:\Thesis\data\MODEL\rf_results_regression\chosen\tree_count_best_params.json"

# --- Load and prep ---
gdf_train = gpd.read_file(train_file)
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
df_train = gdf_train.select_dtypes(include=[np.number])
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]

# --- Define grid ---
param_grid = {
    "n_estimators": [100, 200,300,500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 7,10],
    "min_samples_leaf": [1, 2, 4],               # Min samples at a leaf node
    "max_features": ["sqrt", "log2", 0.3, 0.5],  # Number of features to consider at each split
    "bootstrap": [True, False]  
}

# --- Run grid search ---
print("🔍 Starting hyperparameter tuning...")
grid_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='r2', #change here for regression or classification
    n_jobs=-1,
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

from sklearn.ensemble import RandomForestClassifier  # <- use the classifier here
from sklearn.model_selection import RandomizedSearchCV
import geopandas as gpd
import pandas as pd
import numpy as np
import json

# --- Config ---
target = "binary_recovery"
train_file = r"E:\Thesis\data\shrink_metrics\chosen_pre8020\shrinkmetrics_v10_80_chose.gpkg"
output_path = r"E:\Thesis\data\MODEL\rf_results_regression\chosen\binary_recovery_best_params.json"

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
       'binary_recovery', 'geometry']  # your DEM_only list

# --- Load and prep ---
gdf_train = gpd.read_file(train_file)
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
df_train = gdf_train.select_dtypes(include=[np.number])
X_train = df_train.drop(columns=[target] + exclude_features, errors="ignore")
y_train = df_train[target]

# --- Define grid ---
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 7, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.3, 0.5],
    "bootstrap": [True, False]
}

# --- Run grid search ---
print("🔍 Starting hyperparameter tuning...")

grid_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),  # <--- use classifier here
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='f1',  # <--- classification-appropriate scoring
    n_jobs=-1,
    random_state=42
)
grid_search.fit(X_train, y_train)

print("✅ Grid search complete.")
print("Best params:", grid_search.best_params_)

# --- Save best params ---
with open(output_path, "w") as f:
    json.dump(grid_search.best_params_, f, indent=4)
print(f"Best parameters saved to {output_path}")
