import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

# Convert quality to binary classification
# Good wine (1) if quality >= 7 else Bad wine (0)
df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# 🔁 Cross Validation
cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring="accuracy"
)

print("Cross Validation Accuracy Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

# Train final model
pipeline.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/wine_model.pkl")

print("✅ Model trained and saved successfully")
