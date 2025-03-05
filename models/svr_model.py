# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:51:47 2025

@author: Felix
"""
# svr_model.py
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Data
file_path = r"E:\Thesis\data\metrics_dataset_remaining_80.gpkg"  # Replace with actual file
gdf = gpd.read_file(file_path)

# Define Features (X) and Target (y)
X = gdf.drop(columns=[ 'avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       'plot_tall_veg_%cover', 'plot_forest_%cover', 'side_short_veg_%cover',
       'side_medium_veg_%cover', 'side_tall_veg_%cover', 'side_forest_%cover',
       'adjacency_area_ha', 'adjacency_tree13m_coverage',
       'adjacency_tree8m_coverage', 'adjacency_tree5m_coverage',
       'adjacency_tree3m_coverage', 'adjacency_tree1.5m_coverage',
       'mean_chm', 'mean_chm_under5', 'geometry'], errors="ignore")  # Adjust features
y = gdf["mean_chm"]  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (SVR needs standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR Model
svr_model = SVR(kernel="rbf", C=10, gamma="scale")  # RBF kernel handles non-linearity
svr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svr_model.predict(X_test_scaled)

# Evaluate Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root
r2 = r2_score(y_test, y_pred)

print(f"SVR Model Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# %%

# svr_tuning.py
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
file_path =  r"E:\Thesis\data\metrics_dataset_remaining_80.gpkg"
gdf = gpd.read_file(file_path)

# Define Features (X) and Target (y)
X = gdf.drop(columns=[ 'avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       'plot_tall_veg_%cover', 'plot_forest_%cover', 'side_short_veg_%cover',
       'side_medium_veg_%cover', 'side_tall_veg_%cover', 'side_forest_%cover',
       'adjacency_area_ha', 
       'mean_chm', 'mean_chm_under5', 'geometry'], errors="ignore")  # Adjust features
y = gdf["mean_chm"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (SVR needs normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Hyperparameter Grid
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1]
}

# Perform Grid Search with 5-Fold Cross-Validation
grid_search = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_svr = grid_search.best_estimator_

# Predictions
y_pred = best_svr.predict(X_test_scaled)

# Evaluate Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root
r2 = r2_score(y_test, y_pred)

print(f"Best SVR Parameters: {grid_search.best_params_}")
print(f"Tuned SVR Model Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

#%%
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Compute Feature Importance
result = permutation_importance(best_svr, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring="r2")

# Convert to DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Feature Importance (Permutation)")
plt.ylabel("Feature")
plt.title("SVR Feature Importance")
plt.gca().invert_yaxis()
plt.show()

print(feature_importance_df)
