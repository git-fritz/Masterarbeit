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
file_path = r"E:\Thesis\data\metrics_dataset_remaining_80.gpkg"  # Replace with your actual file path
gdf = gpd.read_file(file_path)

# Display the first few rows
print(gdf.head())

# Check column names
print(gdf.columns)

# %%

# Define the response variable (target)
y = gdf["mean_chm"]  # Replace with the correct column name if different

# List of features to exclude
exclude_features = [ 'avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       'plot_tall_veg_%cover', 'plot_forest_%cover', 'side_short_veg_%cover',
       'side_medium_veg_%cover', 'side_tall_veg_%cover', 'side_forest_%cover',
       'adjacency_area_ha', 'adjacency_tree13m_coverage',
       'adjacency_tree8m_coverage', 'adjacency_tree5m_coverage',
       'adjacency_tree3m_coverage', 'adjacency_tree1.5m_coverage',
       'mean_chm', 'mean_chm_under5', 'geometry']  # Exclude non-predictive columns

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
    max_depth=7,  # Depth of trees (controls complexity)
    subsample=0.8,  # Randomly use 80% of data for each tree (reduces overfitting)
    colsample_bytree=0.8,  # Use 80% of features for each tree
    random_state=42
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
