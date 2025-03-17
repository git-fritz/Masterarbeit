# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:43:56 2025

@author: Felix
"""

# %% model + parameter tuning
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define paths to your GPKG files
train_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_80.gpkg"
vali_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics100_v6_20.gpkg"

# Load data from GPKG
gdf_train = gpd.read_file(train_gpkg_path)
gdf_vali = gpd.read_file(vali_gpkg_path)
print(gdf_train.head())
print(gdf_train.columns)

# Define target variables
target_regression = "median_chm"  # Replace with actual regression target
target_classification = "binary_recovery"  # Replace with actual classification target (0 or 1)

# Define feature exclusion lists
exclude_feature_lists = {
    "DEM_only": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
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
           'binary_recovery', 'geometry'],
    
    "DEM_only+": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
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
           'adjacency_tree1.5m_coverage','mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
           'mean_chm_under2', 'median_chm_under2', 'tree_count', 'tree_density', 'trees_per_ha',
           'plot_pixels', 'veg_pixels_above_60cm', 'veg_cover_percent_above_60cm',
           'binary_recovery', 'geometry'],
}
# %%


# Initialize a results dictionary
results = {"Metric": ["MAE", "RMSE", "R²", "Accuracy", "F1-Score"],
           "Metric_vali": ["MAE_vali", "RMSE_vali", "R²_vali", "Accuracy_vali", "F1-Score_vali"]}

# Drop non-numeric columns
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
gdf_vali = gdf_vali.drop(columns=["geometry"], errors="ignore")

df_train = gdf_train.select_dtypes(include=[np.number])
df_vali = gdf_vali.select_dtypes(include=[np.number])

# Hyperparameter tuning grid for Random Forest
param_grid_reg = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

param_grid_cls = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

# Loop through each exclusion list
for list_name, exclude_features in exclude_feature_lists.items():
    X_train = df_train.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_train_reg = df_train[target_regression]
    y_train_cls = df_train[target_classification]
    
    X_vali = df_vali.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_vali_reg = df_vali[target_regression]
    y_vali_cls = df_vali[target_classification]
    
    # Train Random Forest regression model
    model_reg = GridSearchCV(RandomForestRegressor(), param_grid_reg, cv=3)
    model_reg.fit(X_train, y_train_reg)
    y_pred_reg_train = model_reg.predict(X_train)
    y_pred_reg_vali = model_reg.predict(X_vali)
    
    # Train Random Forest classification model
    model_cls = GridSearchCV(RandomForestClassifier(), param_grid_cls, cv=3)
    model_cls.fit(X_train, y_train_cls)
    y_pred_cls_train = model_cls.predict(X_train)
    y_pred_cls_vali = model_cls.predict(X_vali)
    
    # Calculate training metrics
    mae_train = mean_absolute_error(y_train_reg, y_pred_reg_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_reg, y_pred_reg_train))
    r2_train = r2_score(y_train_reg, y_pred_reg_train)
    accuracy_train = accuracy_score(y_train_cls, y_pred_cls_train)
    f1_train = f1_score(y_train_cls, y_pred_cls_train)
    
    # Calculate validation metrics
    mae_vali = mean_absolute_error(y_vali_reg, y_pred_reg_vali)
    rmse_vali = np.sqrt(mean_squared_error(y_vali_reg, y_pred_reg_vali))
    r2_vali = r2_score(y_vali_reg, y_pred_reg_vali)
    accuracy_vali = accuracy_score(y_vali_cls, y_pred_cls_vali)
    f1_vali = f1_score(y_vali_cls, y_pred_cls_vali)
    
    # Store results
    results[list_name] = [mae_train, rmse_train, r2_train, accuracy_train, f1_train]
    results[f"{list_name}_vali"] = [mae_vali, rmse_vali, r2_vali, accuracy_vali, f1_vali]

# Convert results to DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Save results to CSV
csv_filename = r"E:\Thesis\code_storage\Masterarbeit\results\random_forest_results100_medchm.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")
# %%

# %%

# %% model + parameter tuning
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define paths to your GPKG files
train_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"
vali_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"

# Load data from GPKG
gdf_train = gpd.read_file(train_gpkg_path)
gdf_vali = gpd.read_file(vali_gpkg_path)
print(gdf_train.head())
print(gdf_train.columns)

# Define target variables
target_regression = "tree_count"  # Replace with actual regression target
target_classification = "binary_recovery"  # Replace with actual classification target (0 or 1)

# Define feature exclusion lists
exclude_feature_lists = {
    "DEM_only": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
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
           'binary_recovery', 'geometry'],
    
    "DEM_only+": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
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
           'adjacency_tree1.5m_coverage','mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
           'mean_chm_under2', 'median_chm_under2', 'tree_count', 'tree_density', 'trees_per_ha',
           'plot_pixels', 'veg_pixels_above_60cm', 'veg_cover_percent_above_60cm',
           'binary_recovery', 'geometry'],
}

# Initialize a results dictionary
results = {"Metric": ["MAE", "RMSE", "R²", "Accuracy", "F1-Score"],
           "Metric_vali": ["MAE_vali", "RMSE_vali", "R²_vali", "Accuracy_vali", "F1-Score_vali"]}

# Drop non-numeric columns
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
gdf_vali = gdf_vali.drop(columns=["geometry"], errors="ignore")

df_train = gdf_train.select_dtypes(include=[np.number])
df_vali = gdf_vali.select_dtypes(include=[np.number])

# Hyperparameter tuning grid for Random Forest
param_grid_reg = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "random_state": [42]
}

param_grid_cls = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "random_state": [42]
}

# Loop through each exclusion list
for list_name, exclude_features in exclude_feature_lists.items():
    X_train = df_train.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_train_reg = df_train[target_regression]
    y_train_cls = df_train[target_classification]
    
    X_vali = df_vali.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_vali_reg = df_vali[target_regression]
    y_vali_cls = df_vali[target_classification]
    
    # Train Random Forest regression model
    model_reg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_reg, cv=3)
    model_reg.fit(X_train, y_train_reg)
    y_pred_reg_train = model_reg.predict(X_train)
    y_pred_reg_vali = model_reg.predict(X_vali)
    
    # Train Random Forest classification model
    model_cls = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_cls, cv=3)
    model_cls.fit(X_train, y_train_cls)
    y_pred_cls_train = model_cls.predict(X_train)
    y_pred_cls_vali = model_cls.predict(X_vali)
    
    # Calculate training metrics
    mae_train = mean_absolute_error(y_train_reg, y_pred_reg_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_reg, y_pred_reg_train))
    r2_train = r2_score(y_train_reg, y_pred_reg_train)
    accuracy_train = accuracy_score(y_train_cls, y_pred_cls_train)
    f1_train = f1_score(y_train_cls, y_pred_cls_train)
    
    # Calculate validation metrics
    mae_vali = mean_absolute_error(y_vali_reg, y_pred_reg_vali)
    rmse_vali = np.sqrt(mean_squared_error(y_vali_reg, y_pred_reg_vali))
    r2_vali = r2_score(y_vali_reg, y_pred_reg_vali)
    accuracy_vali = accuracy_score(y_vali_cls, y_pred_cls_vali)
    f1_vali = f1_score(y_vali_cls, y_pred_cls_vali)
    
    # Store results
    results[list_name] = [mae_train, rmse_train, r2_train, accuracy_train, f1_train]
    results[f"{list_name}_vali"] = [mae_vali, rmse_vali, r2_vali, accuracy_vali, f1_vali]

# Convert results to DataFrame
df_results = pd.DataFrame(results)
print(df_results)
       
# Save results to CSV
csv_filename = r"E:\Thesis\code_storage\Masterarbeit\results\random_forest_results_tree.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")
