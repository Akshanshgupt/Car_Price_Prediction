

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ===================== LOAD MODEL =====================
model = pickle.load(open("car_price_model.pkl", "rb"))

# ===================== TITLE =====================
st.title("🚗 Car Price Prediction App")
st.markdown("Predict the **estimated resale price** using **Machine Learning** 🤖")

# ===================== SIDEBAR =====================
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
**Model:** Lasso Regression  
**Features:** 7 (One-Hot Encoded)
""")

st.divider()

# ===================== USER INPUTS =====================
year = st.number_input("📅 Manufacturing Year", 1990, 2025, 2015)
km_driven = st.number_input("🛣️ Kilometers Driven", 0, 500000, 50000)

fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("👤 Seller Type", ["Individual", "Dealer"])
transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

st.divider()

# ===================== SINGLE PREDICTION =====================
if st.button("🔍 Predict Car Price"):

    fuel_CNG = 1 if fuel == "CNG" else 0
    fuel_Diesel = 1 if fuel == "Diesel" else 0
    fuel_Petrol = 1 if fuel == "Petrol" else 0

    seller_type_Individual = 1 if seller_type == "Individual" else 0
    transmission_Manual = 1 if transmission == "Manual" else 0

    X = np.array([[year, km_driven, fuel_CNG, fuel_Diesel, fuel_Petrol,
                   seller_type_Individual, transmission_Manual]])

    price = model.predict(X)[0]
    st.success(f"💰 Estimated Car Price: ₹ {price:,.2f}")

    # ===================== PRICE RANGE GRAPH =====================
    st.subheader("📊 Price Range")
    prices = [price * 0.85, price, price * 1.15]

    fig, ax = plt.subplots()
    ax.bar(["Low", "Estimated", "High"], prices)
    st.pyplot(fig)

# ===================== FEATURE IMPORTANCE =====================
st.subheader("📌 Feature Importance")
features = [
    "Year", "KM Driven", "Fuel CNG",
    "Fuel Diesel", "Fuel Petrol",
    "Seller Individual", "Manual Transmission"
]

importance = np.abs(model.coef_)
fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_title("Lasso Feature Importance")
st.pyplot(fig)

# ===================== MODEL ACCURACY (R²) =====================
st.subheader("📈 Model Accuracy")

st.info("Upload a CSV with true prices to calculate R² score.")

uploaded_r2_file = st.file_uploader("Upload CSV for R² (optional)", type=["csv"], key="r2")

if uploaded_r2_file:
    try:
        df_r2 = pd.read_csv(uploaded_r2_file)
        # Expect columns: all features + 'Actual_Price'
        if "Actual_Price" not in df_r2.columns:
            st.error("CSV must contain 'Actual_Price' column.")
        else:
            X_test = df_r2.drop(columns=["Actual_Price"]).values
            y_true = df_r2["Actual_Price"].values
            y_pred = model.predict(X_test)
            r2 = r2_score(y_true, y_pred)
            st.metric("R² Score", f"{r2:.2f}")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.warning("R² not available without true target values.")

# ===================== CSV UPLOAD & BULK PREDICTION =====================
st.subheader("📂 Upload CSV & Predict")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="bulk")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        predictions = model.predict(df.values)
        df["Predicted_Price"] = predictions
        st.success("✅ Prediction Successful")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error("❌ CSV format mismatch with training data")

# ===================== FOOTER =====================
st.markdown("---")
st.caption("🚀 Built with Streamlit & Machine Learning")
