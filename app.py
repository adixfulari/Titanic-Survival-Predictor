"""
app.py
======
Flask REST API — receives passenger data from HTML,
returns survival probability from trained Random Forest.

Endpoints:
  GET  /          → serves titanic_predictor.html
  POST /predict   → returns { survived, probability, percent }
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML file

# ── Load model artifacts once at startup ─────────────────────────────────────
print("[Flask] Loading model artifacts …")

if not os.path.exists("model/rf_model.pkl"):
    raise FileNotFoundError(
        "Model not found! Run:  python train_model.py  first."
    )

rf_model     = joblib.load("model/rf_model.pkl")
scaler       = joblib.load("model/scaler.pkl")

with open("model/encoder_info.json") as f:
    encoder_info = json.load(f)

# encoder_info["sex"]      → ['female', 'male']   female=0, male=1
# encoder_info["embarked"] → ['C', 'Q', 'S']      C=0, Q=1, S=2
SEX_MAP      = {v: i for i, v in enumerate(encoder_info["sex"])}
EMBARKED_MAP = {v: i for i, v in enumerate(encoder_info["embarked"])}

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "FamilySize", "IsAlone"]

print(f"    SEX_MAP      : {SEX_MAP}")
print(f"    EMBARKED_MAP : {EMBARKED_MAP}")
print("[Flask] Model ready ✅")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the HTML predictor page."""
    return render_template("titanic_predictor.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "pclass":   1 | 2 | 3,
      "sex":      "male" | "female",
      "age":      <number>,
      "sibsp":    <number>,
      "parch":    <number>,
      "fare":     <number>,
      "embarked": "S" | "C" | "Q"
    }

    Returns:
    {
      "survived":    true | false,
      "probability": 0.82,       // 0‒1 float
      "percent":     82           // integer
    }
    """
    try:
        data = request.get_json(force=True)

        # ── Validate & extract ───────────────────────────────────────────────
        pclass   = int(data.get("pclass",   3))
        sex      = str(data.get("sex",      "male")).lower().strip()
        age      = float(data.get("age",    28))
        sibsp    = int(data.get("sibsp",    0))
        parch    = int(data.get("parch",    0))
        fare     = float(data.get("fare",   14.5))
        embarked = str(data.get("embarked", "S")).upper().strip()

        # ── Encode categoricals ───────────────────────────────────────────────
        sex_enc      = SEX_MAP.get(sex, 1)           # default male=1
        embarked_enc = EMBARKED_MAP.get(embarked, 2) # default S=2

        # ── Feature Engineering ───────────────────────────────────────────────
        family_size = sibsp + parch + 1
        is_alone    = int(family_size == 1)

        # ── Build feature DataFrame (maintains column order) ─────────────────
        passenger = pd.DataFrame([{
            "Pclass":     pclass,
            "Sex":        sex_enc,
            "Age":        age,
            "SibSp":      sibsp,
            "Parch":      parch,
            "Fare":       fare,
            "Embarked":   embarked_enc,
            "FamilySize": family_size,
            "IsAlone":    is_alone
        }])[FEATURES]  # enforce column order

        # ── Scale ─────────────────────────────────────────────────────────────
        passenger_scaled = scaler.transform(passenger)

        # ── Predict ───────────────────────────────────────────────────────────
        prob     = float(rf_model.predict_proba(passenger_scaled)[0][1])
        survived = bool(prob >= 0.5)
        percent  = round(prob * 100)

        # ── Log to console ────────────────────────────────────────────────────
        print(f"  Input  : class={pclass}, sex={sex}, age={age}, "
              f"sibsp={sibsp}, parch={parch}, fare={fare}, port={embarked}")
        print(f"  Output : survived={survived}, prob={prob:.4f} ({percent}%)")

        return jsonify({
            "survived":    survived,
            "probability": prob,
            "percent":     percent
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 400


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Titanic Prediction API — Flask Server")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)