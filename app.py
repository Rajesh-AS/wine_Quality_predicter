import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Wine Quality Prediction App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "wine_model.pkl")

# 🔁 Train model if not exists
if not os.path.exists(MODEL_PATH):
    st.warning("Training model for first time...")

    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        sep=";"
    )

    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    st.success("✅ Model trained successfully!")

# Load model
model = joblib.load(MODEL_PATH)
