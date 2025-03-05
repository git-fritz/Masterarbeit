# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:56:29 2025

@author: Felix
"""
# mlp_model.py
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Data
file_path = r"E:\Thesis\data\metrics_dataset_remaining_80.gpkg"
gdf = gpd.read_file(file_path)

# Define Features (X) and Target (y)
X = gdf.drop(columns=[ 'avg_width', 'max_width', 'edge_points', 'plot_id', 'plot_area',
       'plot_short_veg_%cover', 'plot_medium_veg_%cover',
       'plot_tall_veg_%cover', 'plot_forest_%cover', 'side_short_veg_%cover',
       'side_medium_veg_%cover', 'side_tall_veg_%cover', 'side_forest_%cover',
       'adjacency_area_ha',
       'mean_chm', 'mean_chm_under5', 'geometry'], errors="ignore")
y = gdf["mean_chm"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (MLP needs normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP Model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50, 20), activation="relu", solver="adam",
                         max_iter=500, random_state=42, learning_rate_init=0.01)
mlp_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = mlp_model.predict(X_test_scaled)

# Evaluate Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root
r2 = r2_score(y_test, y_pred)

print(f"MLP Model Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

