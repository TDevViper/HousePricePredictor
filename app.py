import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Load trained model and feature columns
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🏠 House Price Predictor")

# 2️⃣ Collect user input
def get_user_input():
    input_data = {}
    # Example numeric inputs (replace/add more as per your dataset)
    input_data['LotArea'] = st.number_input("Lot Area", min_value=0, value=8450)
    input_data['OverallQual'] = st.number_input("Overall Quality", min_value=1, max_value=10, value=7)
    input_data['1stFlrSF'] = st.number_input("1st Floor SF", min_value=0, value=856)
    input_data['2ndFlrSF'] = st.number_input("2nd Floor SF", min_value=0, value=854)
    # Example categorical input
    input_data['MSZoning_RL'] = st.checkbox("MSZoning: RL")  # Example one-hot
    input_data['MSZoning_RM'] = st.checkbox("MSZoning: RM")  # Another option

    return pd.DataFrame([input_data])

input_df = get_user_input()

# 3️⃣ Align input columns with training columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# 4️⃣ Make prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ${prediction:,.2f}")
