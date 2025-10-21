import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and feature columns
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # make sure you have this saved

st.title("🏠 House Price Predictor")

# User-friendly input fields
lot_area = st.number_input("Lot Area (in sq ft)", value=5000)
year_built = st.number_input("Year Built", value=2000)
bedrooms = st.number_input("Number of Bedrooms", value=3)
bathrooms = st.number_input("Number of Bathrooms", value=2)
garage_size = st.number_input("Garage Size (Number of Cars)", value=1)
overall_quality = st.slider("Overall Quality (1-10)", 1, 10, 5)

# Create DataFrame from inputs
input_dict = {
    "LotArea": lot_area,
    "YearBuilt": year_built,
    "BedroomAbvGr": bedrooms,
    "FullBath": bathrooms,
    "GarageCars": garage_size,
    "OverallQual": overall_quality
}
input_df = pd.DataFrame([input_dict])

# Preprocess: add missing columns (from training)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing columns with 0s

# Ensure columns are in the same order as training
input_df = input_df[feature_columns]

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 Predicted House Price: ${prediction:,.2f}")
