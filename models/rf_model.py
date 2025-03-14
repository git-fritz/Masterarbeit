# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:21:36 2025

@author: Felix
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import sklearn
#%%
# Load the dataset
file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_80.gpkg"  # Replace with your actual file path
gdf = gpd.read_file(file_path)

# Display the first few rows
print(gdf.head())

# Check column names
print(gdf.columns)

#%%
# Define the response variable (target)
y = gdf["median_chm"]  # This is what we are predicting

# List of columns to exclude
# exclude_features = ["plot_id", "geometry", "mean_chm", "max_width", "plot_area", "edge_points", 'mean_chm_under5', "plot_forest_%cover", "plot_medium_veg_%cover", "plot_tall_veg_%cover"]  # Exclude ID, geometry, and target

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
       'plot_avg_aspect', 'geometry','hummock_coverage', 'hollow_coverage']
# Select all columns except the excluded ones
X = gdf.drop(columns=exclude_features, errors="ignore")

# Print selected features
print("Selected Features:", X.columns.tolist())

#%%Train-Test Split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset sizes
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

#%%Train the Random Forest Model

# Initialize Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=500,  # Number of trees
    max_depth=10,  # Limits depth to prevent overfitting
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=2,  # Minimum samples per leaf node
    random_state=42,
    n_jobs=-1  # Use all CPU cores for faster training
)

# Train the model
rf_model.fit(X_train, y_train)

print("Model training complete!")

#%%Evaluate Model Performance

# Predict on test data
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# Calculate RMSE manually
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root

r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.3f}")  # Mean Absolute Error
print(f"RMSE: {rmse:.3f}")  # Root Mean Squared Error
print(f"R² Score: {r2:.3f}")  # Variance explained by model

#%% Feature Importance Analysis
# Get feature importances
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title("Top 15 Important Features")
plt.show()

#%%
#%%
#%% Test against unused dataset

# Load the new dataset (replace with your actual file path)
new_file_path = r"E:\Thesis\data\shrink_metrics\auto_pre8020\shrinkmetrics_v6_20.gpkg"
gdf_new = gpd.read_file(new_file_path)

# Check available columns
print("Columns in new dataset:", gdf_new.columns.tolist())

# Preview the first few rows
print(gdf_new.head())
#%%
# Remove non-numeric columns (e.g., ID, geometry, target variable)
# exclude_features = ["plot_id", "geometry", "mean_chm", "max_width", "plot_area", "edge_points", "mean_chm_under5"]  # Exclude ID, geometry, and target

# Select the same features as used for training
X_new = gdf_new.drop(columns=exclude_features, errors="ignore")

# Ensure all columns match the trained model
missing_cols = set(X.columns) - set(X_new.columns)
extra_cols = set(X_new.columns) - set(X.columns)

print(f"Missing columns in new dataset: {missing_cols}")
print(f"Extra columns in new dataset: {extra_cols}")

#%%
# Use the trained model to predict CHM height
y_new_pred = rf_model.predict(X_new)

# Add predictions as a new column to the GeoDataFrame
gdf_new["predicted_chm"] = y_new_pred

# Display first few predictions
print(gdf_new[["predicted_chm"]].head())

#%%

# Check if actual CHM height exists in the new dataset
if "mean_chm" in gdf_new.columns:
    # Compute statistics for actual and predicted CHM height
    stats_comparison = pd.DataFrame({
        "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
        "Actual CHM": [
            gdf_new["median_chm"].mean(),
            gdf_new["median_chm"].median(),
            gdf_new["median_chm"].min(),
            gdf_new["median_chm"].max(),
            gdf_new["median_chm"].std(),
        ],
        "Predicted CHM": [
            gdf_new["predicted_chm"].mean(),
            gdf_new["predicted_chm"].median(),
            gdf_new["predicted_chm"].min(),
            gdf_new["predicted_chm"].max(),
            gdf_new["predicted_chm"].std(),
        ]
    })

    print(stats_comparison)
else:
    print("Actual CHM height (mean_chm) not found in new dataset!")
    
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if "mean_chm" in gdf_new.columns:
    # Compute error metrics
    mae_new = mean_absolute_error(gdf_new["median_chm"], gdf_new["predicted_chm"])
    # rmse_new = mean_squared_error(gdf_new["mean_chm"], gdf_new["predicted_chm"], squared=False)
    rmse_new = mean_squared_error(gdf_new["median_chm"], gdf_new["predicted_chm"]) ** 0.5  # Take the square root

    r2_new = r2_score(gdf_new["median_chm"], gdf_new["predicted_chm"])

    print(f"\nModel Evaluation on New Data:")
    print(f"MAE: {mae_new:.3f}")
    print(f"RMSE: {rmse_new:.3f}")
    print(f"R² Score: {r2_new:.3f}")
else:
    print("Cannot compute error metrics—'mean_chm' column missing in new dataset.")
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt

if "mean_chm" in gdf_new.columns:
    plt.figure(figsize=(10, 5))

    # Plot actual vs predicted CHM distributions
    sns.kdeplot(gdf_new["median_chm"], label="Actual CHM", fill=True, color="blue")
    sns.kdeplot(gdf_new["predicted_chm"], label="Predicted CHM", fill=True, color="red")

    plt.title("Distribution of Actual vs Predicted CHM Heights")
    plt.xlabel("CHM Height")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

#%%
if "mean_chm" in gdf_new.columns:
    plt.figure(figsize=(6,6))

    # Scatter plot
    sns.scatterplot(x=gdf_new["median_chm"], y=gdf_new["predicted_chm"], alpha=0.5)

    # Add a perfect prediction line (y = x)
    plt.plot([gdf_new["median_chm"].min(), gdf_new["median_chm"].max()], 
             [gdf_new["median_chm"].min(), gdf_new["median_chm"].max()], 
             linestyle="--", color="black")

    plt.xlabel("Actual CHM Height")
    plt.ylabel("Predicted CHM Height")
    plt.title("Actual vs Predicted CHM Height")
    plt.show()




# %%






































