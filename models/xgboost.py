# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:16:07 2025

@author: Felix
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
# %%

# Load the dataset
file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"  # Replace with your actual file path
gdf = gpd.read_file(file_path)

# Display the first few rows
print(gdf.head())

# Check column names
print(gdf.columns)

# %%

# Define the response variable (target)
y = gdf["mean_chm_under2"]  # Replace with the correct column name if different

# List of features to exclude
exclude_features = ['id', 'avg_width', 'max_width', 'width_hist', 'width_bins', 'UniqueID',
       'shrink_dis', 'PartID', 'area', 'centroid_x', 'centroid_y',
       'edge_points', 'side', 'SegmentID', 'plot_id', 'segment_area',
       'side_area', 'plot_area', 'plot_short_veg_%cover',
       'plot_medium_veg_%cover', 'plot_tall_veg_%cover', 'plot_forest_%cover',
       'side_short_veg_%cover', 'side_medium_veg_%cover',
       'side_tall_veg_%cover', 'side_forest_%cover',
       'segment_short_veg_%cover', 'segment_medium_veg_%cover',
       'segment_tall_veg_%cover', 'segment_forest_%cover', 'adjacency_area_ha',
       'ndtm_iqr', 'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
       'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
       'plot_avg_aspect', 'geometry']  # Exclude non-predictive columns

# Select all columns except the ones we exclude
X = gdf.drop(columns=exclude_features, errors="ignore")

# Display selected features
print(f"Selected Features ({len(X.columns)}): {X.columns.tolist()}")

# %%

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset sizes
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
# %%

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,  # Number of trees
    learning_rate=0.05,  # Step size
    max_depth=4,  # Depth of trees (controls complexity)
    subsample=0.9,  # Randomly use 80% of data for each tree (reduces overfitting)
    colsample_bytree=0.6,  # Use 80% of features for each tree
    random_state=42, 
    gamma= 0.2,  
    min_child_weight= 7
)

# Train the model
xgb_model.fit(X_train, y_train)

print("Model training complete!")

# %%

# Predict on test data
y_pred = xgb_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.3f}")  # Mean Absolute Error
print(f"RMSE: {rmse:.3f}")  # Root Mean Squared Error
print(f"R² Score: {r2:.3f}")  # Variance explained by model

# %%

# Explain the model using SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test)

# %%

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(xgb_model, X, y, cv=20, scoring="neg_mean_absolute_error")

# Convert negative MAE to positive
cv_mae = -cv_scores

print(f"20-Fold CV Mean MAE: {cv_mae.mean():.3f}")

# %%


# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:16:07 2025

@author: Felix
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap

# %% Load Training Data

# File paths
train_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"  # Training dataset
validation_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"  # Validation dataset

# Load training dataset
gdf_train = gpd.read_file(train_file_path)
print("✅ Training Data Loaded")

# Load validation dataset
gdf_val = gpd.read_file(validation_file_path)
print("✅ Validation Data Loaded")

# %% Feature Selection

# Define the response variable (target)
y_train = gdf_train["mean_chm"]
y_val = gdf_val["mean_chm"]  # Validation target

# Features to exclude
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
       'adjacency_tree1.5m_coverage', 
       'ndtm_iqr', 'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
       'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
       'plot_avg_aspect', 'geometry']

# Select features
X_train = gdf_train.drop(columns=exclude_features, errors="ignore")
X_val = gdf_val.drop(columns=exclude_features, errors="ignore")

# Display feature selection summary
print(f"Training Features ({len(X_train.columns)}): {X_train.columns.tolist()}")
print(f"Validation Features ({len(X_val.columns)}): {X_val.columns.tolist()}")

# %% Model Training

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)
print("✅ Model training complete!")

# %% Performance on Training Data

y_train_pred = xgb_model.predict(X_train)

# Training set metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
train_r2 = r2_score(y_train, y_train_pred)

print("\n📊 Training Data Performance:")
print(f"MAE: {train_mae:.3f}")
print(f"RMSE: {train_rmse:.3f}")
print(f"R² Score: {train_r2:.3f}")

# %% Performance on Validation Data

y_val_pred = xgb_model.predict(X_val)

# Validation set metrics
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
val_r2 = r2_score(y_val, y_val_pred)

print("\n📊 Validation Data Performance:")
print(f"MAE: {val_mae:.3f}")
print(f"RMSE: {val_rmse:.3f}")
print(f"R² Score: {val_r2:.3f}")

# %% SHAP Feature Importance

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_val)

# Plot feature importance
shap.summary_plot(shap_values, X_val)

# %% Cross-Validation

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=20, scoring="neg_mean_absolute_error")

# Convert negative MAE to positive
cv_mae = -cv_scores

print(f"\n📊 20-Fold Cross-Validation Mean MAE: {cv_mae.mean():.3f}")



# %%

# %%

# %%

# %%

# %%

# %%

# %%


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap

# %% Load Training Data

# File paths
train_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"  # Training dataset
validation_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"  # Validation dataset

# Load training dataset
gdf_train = gpd.read_file(train_file_path)
print("✅ Training Data Loaded")

# Load validation dataset
gdf_val = gpd.read_file(validation_file_path)
print("✅ Validation Data Loaded")

# %% Feature Selection

# Define the response variable (target)
y_train = gdf_train["mean_chm"]
y_val = gdf_val["mean_chm"]  # Validation target

# Features to exclude
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
       'adjacency_tree1.5m_coverage', 
       'ndtm_iqr', 'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
       'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
       'plot_avg_aspect', 'geometry']

# Select features
X_train = gdf_train.drop(columns=exclude_features, errors="ignore")
X_val = gdf_val.drop(columns=exclude_features, errors="ignore")

# Ensure feature consistency
missing_cols = set(X_train.columns) - set(X_val.columns)
extra_cols = set(X_val.columns) - set(X_train.columns)

print(f"Missing columns in validation dataset: {missing_cols}")
print(f"Extra columns in validation dataset: {extra_cols}")

# %% Model Training

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)
print("✅ Model training complete!")
# %%

from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {
    'max_depth': [4, 5, 6, 7],
    'min_child_weight': [1, 3, 5, 7]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(
        n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)

# %% Performance on Training Data

y_train_pred = xgb_model.predict(X_train)

# Training set metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
train_r2 = r2_score(y_train, y_train_pred)

print("\n📊 Training Data Performance:")
print(f"MAE: {train_mae:.3f}")
print(f"RMSE: {train_rmse:.3f}")
print(f"R² Score: {train_r2:.3f}")

# %% Performance on Validation Data

y_val_pred = xgb_model.predict(X_val)

# Validation set metrics
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
val_r2 = r2_score(y_val, y_val_pred)

print("\n📊 Validation Data Performance:")
print(f"MAE: {val_mae:.3f}")
print(f"RMSE: {val_rmse:.3f}")
print(f"R² Score: {val_r2:.3f}")

# %% SHAP Feature Importance

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_val)

# Plot feature importance
shap.summary_plot(shap_values, X_val)

# %% Cross-Validation

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=20, scoring="neg_mean_absolute_error")

# Convert negative MAE to positive
cv_mae = -cv_scores

print(f"\n📊 20-Fold Cross-Validation Mean MAE: {cv_mae.mean():.3f}")

# %% Save Predictions to GeoPackage
# gdf_val["predicted_chm"] = y_val_pred

# # Save the updated validation dataset with predictions
# output_gpkg = r"E:\Thesis\data\shrink_metrics\validation_predictions.gpkg"
# gdf_val.to_file(output_gpkg, driver="GPKG")

# print(f"\n✅ Updated validation dataset saved with predictions: {output_gpkg}")
# %%

# %%

# %%

# %%

# %%

# %% Load Required Libraries
import geopandas as gpd
import numpy as np
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% Load Training & Validation Data
train_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"
validation_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"

gdf_train = gpd.read_file(train_file_path)
gdf_val = gpd.read_file(validation_file_path)

print("✅ Training & Validation Data Loaded")

# %% Feature Selection

y_train = gdf_train["mean_chm"]
y_val = gdf_val["mean_chm"]

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
       'adjacency_tree1.5m_coverage', 
       'ndtm_iqr', 'mean_chm', 'median_chm', 'mean_chm_under5', 'median_chm_under5',
       'mean_chm_under2', 'median_chm_under2', 'plot_avg_slope',
       'plot_avg_aspect', 'geometry']

X_train = gdf_train.drop(columns=exclude_features, errors="ignore")
X_val = gdf_val.drop(columns=exclude_features, errors="ignore")

# Ensure feature consistency
missing_cols = set(X_train.columns) - set(X_val.columns)
extra_cols = set(X_val.columns) - set(X_train.columns)

print(f"Missing columns in validation dataset: {missing_cols}")
print(f"Extra columns in validation dataset: {extra_cols}")

# %% Model Training with Early Stopping
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "validation")],
    early_stopping_rounds=20,
    verbose_eval=True
)

print("✅ Model training complete!")

# %% Hyperparameter Tuning with Grid Search
param_grid = {
    'max_depth': [4, 5, 6, 7],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(
        n_estimators=500, learning_rate=0.05, random_state=42
    ),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best parameters found
best_params = grid_search.best_params_
print("🔍 Best Hyperparameters:", best_params)

# %% Retrain with Best Parameters
xgb_best = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    **best_params,
    random_state=42
)

# xgb_best.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     eval_metric="rmse",
#     early_stopping_rounds=20,
#     verbose=True
# )


print("✅ Best model retrained!")

# %% Performance on Training Data
y_train_pred = xgb_best.predict(X_train)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
train_r2 = r2_score(y_train, y_train_pred)

print("\n📊 Training Data Performance:")
print(f"MAE: {train_mae:.3f}")
print(f"RMSE: {train_rmse:.3f}")
print(f"R² Score: {train_r2:.3f}")

# %% Performance on Validation Data
y_val_pred = xgb_best.predict(X_val)

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
val_r2 = r2_score(y_val, y_val_pred)

print("\n📊 Validation Data Performance:")
print(f"MAE: {val_mae:.3f}")
print(f"RMSE: {val_rmse:.3f}")
print(f"R² Score: {val_r2:.3f}")

# %% SHAP Feature Importance (Optimized)
explainer = shap.Explainer(xgb_best)
shap_values = explainer.shap_values(X_val.sample(500))  # Use a sample to speed up computation

shap.summary_plot(shap_values, X_val.sample(500))

# %% Cross-Validation
cv_scores = cross_val_score(xgb_best, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

cv_mae = -cv_scores
print(f"\n📊 5-Fold Cross-Validation Mean MAE: {cv_mae.mean():.3f}")


