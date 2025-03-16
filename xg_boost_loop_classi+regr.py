# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:56:27 2025

@author: Felix
"""
# %% model clean 


import geopandas as gpd
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Define paths to your GPKG files
train_gpkg_path = "train_data.gpkg"   # Replace with your actual training dataset path
vali_gpkg_path = "validation_data.gpkg"  # Replace with your actual validation dataset path

# Load data from GPKG
gdf_train = gpd.read_file(train_gpkg_path)
gdf_vali = gpd.read_file(vali_gpkg_path)

# Define target variables
target_regression = "seedling_count"  # Change to your actual regression target
target_classification = "recovery_status"  # Change to your classification target (0 or 1)

# Define feature exclusion lists
exclude_feature_lists = {
    "List1": ["feature_A", "feature_B"],
    "List2": ["feature_C", "feature_D"],
    "List3": ["feature_E", "feature_F"],
}

# Initialize a results dictionary
results = {"Metric": ["MAE", "RMSE", "R²", "Accuracy", "F1-Score"],
           "Metric_vali": ["MAE_vali", "RMSE_vali", "R²_vali", "Accuracy_vali", "F1-Score_vali"]}

# Convert to DataFrame and drop non-numeric columns
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")  # Remove geometry for ML
gdf_vali = gdf_vali.drop(columns=["geometry"], errors="ignore")

df_train = gdf_train.select_dtypes(include=[np.number])  # Keep only numeric columns
df_vali = gdf_vali.select_dtypes(include=[np.number])

# Loop through each exclusion list
for list_name, exclude_features in exclude_feature_lists.items():
    # Prepare feature sets for training
    X_train = df_train.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_train_reg = df_train[target_regression]
    y_train_cls = df_train[target_classification]

    # Prepare feature sets for validation
    X_vali = df_vali.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_vali_reg = df_vali[target_regression]
    y_vali_cls = df_vali[target_classification]

    # Train regression model
    model_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model_reg.fit(X_train, y_train_reg)
    y_pred_reg_train = model_reg.predict(X_train)
    y_pred_reg_vali = model_reg.predict(X_vali)

    # Train classification model
    model_cls = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100)
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

    # Store results in dictionary
    results[list_name] = [mae_train, rmse_train, r2_train, accuracy_train, f1_train]
    results[f"{list_name}_vali"] = [mae_vali, rmse_vali, r2_vali, accuracy_vali, f1_vali]

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Model Training & Validation Results", dataframe=df_results)

# %% model + parameter tuning
import geopandas as gpd
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Define paths to your GPKG files
train_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"
vali_gpkg_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"

# Load data from GPKG
gdf_train = gpd.read_file(train_gpkg_path)
gdf_vali = gpd.read_file(vali_gpkg_path)
print(gdf_train.head)
print(gdf_train.columns)

# %%


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
    
    "veg_adjacency": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
           'shrink_dis', 'PartID', 'area', 'centroid_x', 'centroid_y',
           'edge_points', 'side', 'SegmentID', 'plot_id', 'segment_area',
           'side_area', 'plot_area', 'plot_short_veg_%cover',
           'plot_medium_veg_%cover', 'plot_tall_veg_%cover', 'plot_forest_%cover',
           'side_short_veg_%cover', 'side_medium_veg_%cover',
           'side_tall_veg_%cover', 'side_forest_%cover', 'hummock_coverage', 'hollow_coverage',
           'ndtm_iqr', 'dem_diff', 'dem_SD', 'rough_avg', 'rough_SD',
           'mound_count_15percentile', 'mound_area_15percentile',
           'mound_density_15percentile', 'mound_coverage_15percentile',
           'max_mound_size_15percentile', 'avg_mound_size_15percentile',
           'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
           'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
           'plot_avg_aspect', 'tree_count', 'tree_density', 'trees_per_ha',
           'plot_pixels', 'veg_pixels_above_60cm', 'veg_cover_percent_above_60cm',
           'binary_recovery', 'geometry','segment_short_veg_%cover', 'segment_medium_veg_%cover',
           'segment_tall_veg_%cover', 'segment_forest_%cover', 'adjacency_area_ha',],
    
    "DEM_only+_veg_adjacency": ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
           'shrink_dis', 'PartID', 'area', 'centroid_x', 'centroid_y',
           'edge_points', 'side', 'SegmentID', 'plot_id', 'segment_area',
           'side_area', 'plot_area', 'plot_short_veg_%cover',
           'plot_medium_veg_%cover', 'plot_tall_veg_%cover', 'plot_forest_%cover',
           'side_short_veg_%cover', 'side_medium_veg_%cover',
           'side_tall_veg_%cover', 'side_forest_%cover', 'hummock_coverage', 'hollow_coverage',
           'ndtm_iqr', 'dem_diff', 'dem_SD', 'rough_avg', 'rough_SD',
           'mound_count_15percentile', 'mound_area_15percentile',
           'mound_density_15percentile', 'mound_coverage_15percentile',
           'max_mound_size_15percentile', 'avg_mound_size_15percentile',
           'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
           'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
           'plot_avg_aspect', 'tree_count', 'tree_density', 'trees_per_ha',
           'plot_pixels', 'veg_pixels_above_60cm', 'veg_cover_percent_above_60cm',
           'binary_recovery', 'geometry','segment_short_veg_%cover', 'segment_medium_veg_%cover',
           'segment_tall_veg_%cover', 'segment_forest_%cover', 'adjacency_area_ha',]
}
# %% feature importance
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# from sklearn.feature_selection import VarianceThreshold, RFE
# from sklearn.model_selection import train_test_split

# # Ensure the dataset is numeric
# X_full = df_train.drop(columns=[target_regression, target_classification], errors="ignore")
# y_full_reg = df_train[target_regression]

# # Train a baseline XGBoost model to analyze feature importance
# model_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
# model_reg.fit(X_full, y_full_reg)

# # 🎯 1. Plot Feature Importance (XGBoost)
# plt.figure(figsize=(10, 6))
# xgb.plot_importance(model_reg, max_num_features=20)  # Show top 20 features
# plt.title("XGBoost Feature Importance (Regression)")
# plt.show()

# # 🎯 2. Correlation Analysis (Remove Highly Correlated Features)
# corr_matrix = X_full.corr()
# high_corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
# high_corr_pairs = high_corr_pairs[high_corr_pairs != 1]  # Exclude self-correlation

# print("Highly Correlated Feature Pairs:")
# print(high_corr_pairs[:20])

# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()

# # 🎯 3. Variance Thresholding (Remove Low-Variance Features)
# selector = VarianceThreshold(threshold=0.01)  # Adjust threshold if needed
# X_reduced = selector.fit_transform(X_full)

# kept_features = X_full.columns[selector.get_support()]
# removed_features = X_full.columns[~selector.get_support()]
# print(f"Removed features due to low variance: {list(removed_features)}")

# # 🎯 4. Recursive Feature Elimination (RFE)
# rfe = RFE(model_reg, n_features_to_select=10)  # Adjust n_features_to_select
# rfe.fit(X_full, y_full_reg)

# selected_features = X_full.columns[rfe.support_]
# removed_features_rfe = X_full.columns[~rfe.support_]
# print(f"Selected features (RFE): {list(selected_features)}")
# print(f"Removed features (RFE): {list(removed_features_rfe)}")

# %%
# Initialize a results dictionary
results = {"Metric": ["MAE", "RMSE", "R²", "Accuracy", "F1-Score"],
           "Metric_vali": ["MAE_vali", "RMSE_vali", "R²_vali", "Accuracy_vali", "F1-Score_vali"]}

# Drop non-numeric columns (like geometry) from datasets
gdf_train = gdf_train.drop(columns=["geometry"], errors="ignore")
gdf_vali = gdf_vali.drop(columns=["geometry"], errors="ignore")

df_train = gdf_train.select_dtypes(include=[np.number])  # Keep only numeric columns
df_vali = gdf_vali.select_dtypes(include=[np.number])

# Hyperparameter tuning grid for XGBoost
param_grid_reg = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

param_grid_cls = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

# Loop through each exclusion list
for list_name, exclude_features in exclude_feature_lists.items():
    # Prepare feature sets for training
    X_train = df_train.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_train_reg = df_train[target_regression]
    y_train_cls = df_train[target_classification]

    # Prepare feature sets for validation
    X_vali = df_vali.drop(columns=[target_regression, target_classification] + exclude_features, errors="ignore")
    y_vali_reg = df_vali[target_regression]
    y_vali_cls = df_vali[target_classification]

    # Train regression model with hyperparameter tuning
    model_reg = GridSearchCV(xgb.XGBRegressor(objective="reg:squarederror"), param_grid_reg, cv=3)
    model_reg.fit(X_train, y_train_reg)
    y_pred_reg_train = model_reg.predict(X_train)
    y_pred_reg_vali = model_reg.predict(X_vali)

    # Train classification model with hyperparameter tuning
    model_cls = GridSearchCV(xgb.XGBClassifier(objective="binary:logistic"), param_grid_cls, cv=3)
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

    # Store results in dictionary
    results[list_name] = [mae_train, rmse_train, r2_train, accuracy_train, f1_train]
    results[f"{list_name}_vali"] = [mae_vali, rmse_vali, r2_vali, accuracy_vali, f1_vali]

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Print results in console
print(df_results)

# Save results to CSV
csv_filename = r"E:\Thesis\code_storage\Masterarbeit\results\xgboost_results_v1.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

