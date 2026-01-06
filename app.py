import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Wine Quality Prediction App")
st.write("Predict wine quality using machine learning")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "wine_model.pkl")

# ===============================
# TRAIN MODEL IF NOT EXISTS
# ===============================
if not os.path.exists(MODEL_PATH):
    st.warning("⏳ Training model for the first time...")

    # Load dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    # Binary classification
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

    X = df.drop("quality", axis=1)
    y = df["quality"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    st.success("✅ Model trained successfully!")

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load(MODEL_PATH)

# ===============================
# USER INPUTS
# ===============================
st.subheader("Enter Wine Chemical Properties")

fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.076)
free_sulfur = st.number_input("Free Sulfur Dioxide", 1, 80, 11)
total_sulfur = st.number_input("Total Sulfur Dioxide", 6, 300, 34)
density = st.number_input("Density", 0.9900, 1.0050, 0.9978, format="%.4f")
ph = st.number_input("pH", 2.5, 4.0, 3.51)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, 9.4)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur,
                             total_sulfur, density, ph, sulphates, alcohol]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Bad Quality Wine")
