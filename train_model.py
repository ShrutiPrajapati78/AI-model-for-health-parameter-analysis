"""
train_model.py
Trains a Random Forest classifier to predict health status
(Normal / At Risk / Critical) from:
    - systolic_bp
    - diastolic_bp
    - temperature_f
    - spo2

Saves the trained model + scaler to disk using joblib.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

FEATURES = ["systolic_bp", "diastolic_bp", "temperature_f", "spo2"]
TARGET = "health_status"


def load_data(path="health_data.csv"):
    return pd.read_csv(path)


def train():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest — main model
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train_scaled, y_train)

    # Logistic Regression — quick baseline for comparison
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)

    for name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print("Accuracy:", round(acc, 4))
        print(classification_report(y_test, preds, target_names=le.classes_))

    # feature importance (RF)
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances (Random Forest):")
    print(importances)

    # Save best model (Random Forest), scaler, and label encoder
    joblib.dump(rf, "health_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(le, "label_encoder.joblib")
    print("\nSaved: health_model.joblib, scaler.joblib, label_encoder.joblib")


if __name__ == "__main__":
    train()
