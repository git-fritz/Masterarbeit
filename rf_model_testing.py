# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 19:27:57 2025

@author: Felix
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 19:17:26 2025

@author: Felix
"""

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
file_path = r"E:\Thesis\data\metrics_dataset_remaining_80.gpkg"  # Replace with your actual file path
gdf = gpd.read_file(file_path)

# Display the first few rows
print(gdf.head())

# Check column names
print(gdf.columns)

#%%
# Define the response variable (target)
y = gdf["mean_chm_under5"]  # This is what we are predicting

# List of columns to exclude
exclude_features = ['avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       'plot_tall_veg_%cover', 'plot_forest_%cover', 'mean_chm', 'mean_chm_under5', 'geometry']  # Exclude ID, geometry, and target

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
    max_depth=20,  # Limits depth to prevent overfitting
    min_samples_split=10,  # Minimum samples required to split a node
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

# %%

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="neg_mean_absolute_error")

# Convert negative MAE to positive
cv_mae = -cv_scores

print(f"5-Fold CV Mean MAE: {cv_mae.mean():.3f}")

#%%
#%%
#%% Test against unused dataset

# Load the new dataset (replace with your actual file path)
new_file_path = r"E:\Thesis\data\metrics_dataset_subset_20.gpkg" 
gdf_new = gpd.read_file(new_file_path)

# Check available columns
print("Columns in new dataset:", gdf_new.columns.tolist())

# Preview the first few rows
print(gdf_new.head())
#%%
# Remove non-numeric columns (e.g., ID, geometry, target variable)
# exclude_features = ["plot_id", "geometry", "mean_chm", 'avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       # 'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       # 'plot_tall_veg_%cover', 'plot_forest_%cover', 'side_short_veg_%cover',
       # 'side_medium_veg_%cover', 'side_tall_veg_%cover', 'side_forest_%cover',
       # 'adjacency_area_ha', 'adjacency_tree13m_coverage',
       # 'adjacency_tree8m_coverage', 'adjacency_tree5m_coverage',
       # 'adjacency_tree3m_coverage', 'adjacency_tree1.5m_coverage', 'mean_chm_under5']  # Exclude ID, geometry, and target

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
if "mean_chm_under5" in gdf_new.columns:
    # Compute statistics for actual and predicted CHM height
    stats_comparison = pd.DataFrame({
        "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
        "Actual CHM": [
            gdf_new["mean_chm_under5"].mean(),
            gdf_new["mean_chm_under5"].median(),
            gdf_new["mean_chm_under5"].min(),
            gdf_new["mean_chm_under5"].max(),
            gdf_new["mean_chm_under5"].std(),
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
    print("Actual CHM height (mean_chm_under5) not found in new dataset!")
    
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if "mean_chm_under5" in gdf_new.columns:
    # Compute error metrics
    mae_new = mean_absolute_error(gdf_new["mean_chm_under5"], gdf_new["predicted_chm"])
    # rmse_new = mean_squared_error(gdf_new["mean_chm"], gdf_new["predicted_chm"], squared=False)
    rmse_new = mean_squared_error(gdf_new["mean_chm_under5"], gdf_new["predicted_chm"]) ** 0.5  # Take the square root

    r2_new = r2_score(gdf_new["mean_chm_under5"], gdf_new["predicted_chm"])

    print(f"\nModel Evaluation on New Data:")
    print(f"MAE: {mae_new:.3f}")
    print(f"RMSE: {rmse_new:.3f}")
    print(f"R² Score: {r2_new:.3f}")
else:
    print("Cannot compute error metrics—'mean_chm' column missing in new dataset.")
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt

if "mean_chm_under5" in gdf_new.columns:
    plt.figure(figsize=(10, 5))

    # Plot actual vs predicted CHM distributions
    sns.kdeplot(gdf_new["mean_chm_under5"], label="Actual CHM", fill=True, color="blue")
    sns.kdeplot(gdf_new["predicted_chm"], label="Predicted CHM", fill=True, color="red")

    plt.title("Distribution of Actual vs Predicted CHM Heights")
    plt.xlabel("CHM Height")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

#%%
if "mean_chm_under5" in gdf_new.columns:
    plt.figure(figsize=(6,6))

    # Scatter plot
    sns.scatterplot(x=gdf_new["mean_chm_under5"], y=gdf_new["predicted_chm"], alpha=0.5)

    # Add a perfect prediction line (y = x)
    plt.plot([gdf_new["mean_chm_under5"].min(), gdf_new["mean_chm_under5"].max()], 
             [gdf_new["mean_chm_under5"].min(), gdf_new["mean_chm_under5"].max()], 
             linestyle="--", color="black")

    plt.xlabel("Actual CHM Height")
    plt.ylabel("Predicted CHM Height")
    plt.title("Actual vs Predicted CHM Height")
    plt.show()


# %%
#%%
# %%

# %%

# %%


from sklearn.model_selection import RandomizedSearchCV

# Define the parameter distribution
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],  
    'max_depth': [10, 20, 30, None],  
    'min_samples_split': [2, 5, 10, 15],  
    'min_samples_leaf': [1, 2, 4, 8],  
    'max_features': ['sqrt', 'log2', None]  
}

# Create the model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Perform Randomized Search with 5-fold cross-validation
random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, 
    n_iter=20, cv=5, scoring='neg_mean_absolute_error', 
    n_jobs=-1, verbose=2, random_state=42
)

# Fit the search
random_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", random_search.best_params_)

# %%

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define different tree counts to test
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 500]

# Store results
train_r2 = []
test_r2 = []
train_rmse = []
test_rmse = []
train_mae = []
test_mae = []

# Iterate over different numbers of trees
for estimator in n_estimators:
    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)  # Train the model
    
    # Predict on training set
    y_train_pred = rf.predict(X_train)
    
    # Predict on test set
    y_test_pred = rf.predict(X_test)
    
    # Compute R² Score
    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))
    
    # Compute RMSE
    train_rmse.append(mean_squared_error(y_train, y_train_pred) ** 0.5)  # Take square root
    test_rmse.append(mean_squared_error(y_test, y_test_pred) ** 0.5)  # Take square root

    # Compute MAE
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))

# Plot results
plt.figure(figsize=(12, 6))

# Plot R² Score
plt.subplot(1, 2, 1)
plt.plot(n_estimators, train_r2, 'b', label="Train R²")
plt.plot(n_estimators, test_r2, 'r', label="Test R²")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("R² Score")
plt.title("R² Score vs. Number of Trees")
plt.legend()

# Plot RMSE
plt.subplot(1, 2, 2)
plt.plot(n_estimators, train_rmse, 'b', label="Train RMSE")
plt.plot(n_estimators, test_rmse, 'r', label="Test RMSE")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("RMSE")
plt.title("RMSE vs. Number of Trees")
plt.legend()

plt.tight_layout()
plt.show()

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

max_depth_values = [3, 5, 10, 15, 20, 30, None]  # None means unlimited depth

train_r2 = []
test_r2 = []
train_rmse = []
test_rmse = []

for depth in max_depth_values:
    rf = RandomForestRegressor(n_estimators=300, max_depth=depth, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse.append(mean_squared_error(y_train, y_train_pred) ** 0.5)  # Take square root
    test_rmse.append(mean_squared_error(y_test, y_test_pred) ** 0.5)  # Take square root

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(max_depth_values[:-1], train_r2[:-1], 'b', label="Train R²")
plt.plot(max_depth_values[:-1], test_r2[:-1], 'r', label="Test R²")
plt.xlabel("Max Depth")
plt.ylabel("R² Score")
plt.title("R² Score vs. Max Depth")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(max_depth_values[:-1], train_rmse[:-1], 'b', label="Train RMSE")
plt.plot(max_depth_values[:-1], test_rmse[:-1], 'r', label="Test RMSE")
plt.xlabel("Max Depth")
plt.ylabel("RMSE")
plt.title("RMSE vs. Max Depth")
plt.legend()

plt.tight_layout()
plt.show()

# %%

split_values = [2, 5, 10, 15, 20, 30]

train_r2 = []
test_r2 = []
train_rmse = []
test_rmse = []

for split in split_values:
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=split, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse.append(mean_squared_error(y_train, y_train_pred) ** 0.5)  # Take square root
    test_rmse.append(mean_squared_error(y_test, y_test_pred) ** 0.5)  # Take square root

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(split_values, train_r2, 'b', label="Train R²")
plt.plot(split_values, test_r2, 'r', label="Test R²")
plt.xlabel("Min Samples Split")
plt.ylabel("R² Score")
plt.title("R² Score vs. Min Samples Split")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(split_values, train_rmse, 'b', label="Train RMSE")
plt.plot(split_values, test_rmse, 'r', label="Test RMSE")
plt.xlabel("Min Samples Split")
plt.ylabel("RMSE")
plt.title("RMSE vs. Min Samples Split")
plt.legend()

plt.tight_layout()
plt.show()

# %%

leaf_values = [1, 2, 4, 8, 16, 32]

train_r2 = []
test_r2 = []
train_rmse = []
test_rmse = []

for leaf in leaf_values:
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=15, min_samples_leaf=leaf, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse.append(mean_squared_error(y_train, y_train_pred) ** 0.5)  # Take square root
    test_rmse.append(mean_squared_error(y_test, y_test_pred) ** 0.5)  # Take square root

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(leaf_values, train_r2, 'b', label="Train R²")
plt.plot(leaf_values, test_r2, 'r', label="Test R²")
plt.xlabel("Min Samples Leaf")
plt.ylabel("R² Score")
plt.title("R² Score vs. Min Samples Leaf")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(leaf_values, train_rmse, 'b', label="Train RMSE")
plt.plot(leaf_values, test_rmse, 'r', label="Test RMSE")
plt.xlabel("Min Samples Leaf")
plt.ylabel("RMSE")
plt.title("RMSE vs. Min Samples Leaf")
plt.legend()

plt.tight_layout()
plt.show()

# %%

results_df = pd.DataFrame({
    "Metric": ["Train R²", "Test R²", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE"],
    "Value": [r2_score(y_train, rf_model.predict(X_train)),
              r2_score(y_test, rf_model.predict(X_test)),
              mean_squared_error(y_train, rf_model.predict(X_train)) ** 0.5,
              mean_squared_error(y_test, rf_model.predict(X_test)) ** 0.5,
              mean_absolute_error(y_train, rf_model.predict(X_train)),
              mean_absolute_error(y_test, rf_model.predict(X_test))]
})

# Display the results in a readable format
print(results_df)






































