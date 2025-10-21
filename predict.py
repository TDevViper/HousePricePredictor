import joblib
import pandas as pd

# Load model
model = joblib.load("best_model.pkl")

# Example input (replace with real data later)
data = {
    "OverallQual": [7],
    "GrLivArea": [1710],
    "GarageCars": [2],
    "TotalBsmtSF": [856],
    "FullBath": [2],
    "YearBuilt": [2003]
}

df = pd.DataFrame(data)

# Predict
prediction = model.predict(df)
print(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")
