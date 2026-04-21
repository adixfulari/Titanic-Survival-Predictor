"""
train_model.py
==============
Trains Random Forest on Kaggle train.csv and saves:
  model/rf_model.pkl
  model/scaler.pkl
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("[1] Loading train.csv …")
df = pd.read_csv("train.csv")
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(df.head())

# ── 2. Drop Irrelevant Columns ────────────────────────────────────────────────
# PassengerId, Name, Ticket, Cabin are not useful features
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 3. Handle Missing Values ──────────────────────────────────────────────────
print("\n[2] Missing values:")
print(df.isnull().sum())

# Age: fill with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked: fill with mode (only 2 missing in Kaggle dataset)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Fare: fill with median (just in case)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

print("\n    After cleaning:")
print(df.isnull().sum())

# ── 4. Encode Categorical Variables ──────────────────────────────────────────
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])          # female=0, male=1

le_emb = LabelEncoder()
df["Embarked"] = le_emb.fit_transform(df["Embarked"]) # C=0, Q=1, S=2

# Save the label encoder classes so we can use them in Flask
print(f"\n    Sex classes      : {le_sex.classes_}")   # ['female' 'male']
print(f"    Embarked classes : {le_emb.classes_}")   # ['C' 'Q' 'S']

# ── 5. Feature Engineering ────────────────────────────────────────────────────
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

# ── 6. Define Features & Target ───────────────────────────────────────────────
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "FamilySize", "IsAlone"]
TARGET   = "Survived"

X = df[FEATURES]
y = df[TARGET]

# ── 7. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── 8. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 9. Train Random Forest ────────────────────────────────────────────────────
print("\n[4] Training Random Forest Classifier …")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# ── 10. Evaluate ──────────────────────────────────────────────────────────────
y_pred = rf_model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
print(f"\n    ✅  Test Accuracy : {acc * 100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['Did Not Survive','Survived'])}")

# ── 11. Save Model & Scaler ───────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/rf_model.pkl")
joblib.dump(scaler,   "model/scaler.pkl")

# Save encoder mapping so Flask knows how to encode user input
import json
encoder_info = {
    "sex":      list(le_sex.classes_),      # index = encoded value
    "embarked": list(le_emb.classes_)       # index = encoded value
}
with open("model/encoder_info.json", "w") as f:
    json.dump(encoder_info, f)

print("\n[5] Saved:")
print("    model/rf_model.pkl")
print("    model/scaler.pkl")
print("    model/encoder_info.json")
print("\n✅  Training complete!")