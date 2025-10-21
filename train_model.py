# ==========================================

# 🏠 House Price Prediction - ML Project

# Author: Arnav Yadav

# ==========================================

# ✅ Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ==========================================

# Step 1: Load Dataset

# ==========================================

df = pd.read_csv("train.csv")
print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ==========================================

# Step 2: Handle Missing Values

# ==========================================

# Fill numerical columns with median

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode

cat_cols = df.select_dtypes(include=['object']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print("✅ Missing Values Handled!")

# ==========================================

# Step 3: Encode Categorical Columns

# ==========================================

df = pd.get_dummies(df, drop_first=True)
print("✅ Categorical Columns Encoded!")
print("New Shape after encoding:", df.shape)

# ==========================================

# Step 4: Define Features and Target

# ==========================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# ==========================================

# Step 5: Train-Test Split

# ==========================================

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data Split Done!")
print("Train Shape:", x_train.shape)
print("Test Shape:", x_test.shape)

# ==========================================

# Step 6: Define Models

# ==========================================

models = {
"Linear Regression": LinearRegression(),
"Ridge": Ridge(),
"Lasso": Lasso(),
"ElasticNet": ElasticNet(),
"Decision Tree": DecisionTreeRegressor(),
"Random Forest": RandomForestRegressor(),
"Gradient Boosting": GradientBoostingRegressor()
}

# ==========================================

# Step 7: Train and Evaluate Models

# ==========================================

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}


# Convert results to DataFrame

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="RMSE")
print("\n📊 Model Performance Comparison:\n")
print(results_df)

# ==========================================

# Step 8: Save Best Model

# ==========================================

best_model_name = results_df.index[0]
best_model = models[best_model_name]

# Train again on full data

best_model.fit(X, y)
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.pkl")
print(f"\n💾 Best Model '{best_model_name}' Saved Successfully!")

import joblib

# After preprocessing, encoding, and creating X (features)
feature_columns = X.columns  # Save all feature names
joblib.dump(feature_columns, "feature_columns.pkl")
print("✅ Feature columns saved successfully!")


# ==========================================

# END OF SCRIPT

# ==========================================
