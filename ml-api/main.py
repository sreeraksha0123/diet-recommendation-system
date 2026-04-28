"""
Phase 2: FastAPI ML API
Loads trained model artifacts and exposes POST /predict endpoint.

Run with:  uvicorn main:app --reload --port 8000
Test at:   http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
import json

from preprocessing import preprocess_input

app = FastAPI(title="Diet Recommendation ML API", version="1.0")

# ─────────────────────────────────────────
# Load model artifacts on startup
# ─────────────────────────────────────────
MODEL_DIR = "model"

with open(f"{MODEL_DIR}/mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{MODEL_DIR}/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open(f"{MODEL_DIR}/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open(f"{MODEL_DIR}/feature_config.json", "r") as f:
    feature_config = json.load(f)

print(f"Model loaded successfully.")
print(f"Target classes: {feature_config['target_classes']}")
print(f"Training accuracy: {feature_config['fuzzy_mlp_accuracy'] * 100:.2f}%")


# ─────────────────────────────────────────
# Request schema (what Node.js backend will send)
# ─────────────────────────────────────────
class UserInput(BaseModel):
    Age: float
    Gender: str                                      # "Male" or "Female"
    Weight_kg: float
    Height_cm: float
    BMI: float
    Disease_Type: Optional[str] = "None"             # "Diabetes" / "Hypertension" / "Obesity" / "None"
    Severity: str                                    # "Mild" / "Moderate" / "Severe"
    Physical_Activity_Level: str                     # "Sedentary" / "Moderate" / "Active"
    Daily_Caloric_Intake: float
    Cholesterol_mgdL: float                          # Node sends as Cholesterol_mgdL
    Blood_Pressure_mmHg: float
    Glucose_mgdL: float                              # Node sends as Glucose_mgdL
    Dietary_Restrictions: Optional[str] = "None"    # "Low_Sugar" / "Low_Sodium" / "None"
    Allergies: Optional[str] = "None"               # "Gluten" / "Peanuts" / "None"
    Preferred_Cuisine: str                           # "Indian" / "Chinese" / "Italian" / "Mexican"
    Weekly_Exercise_Hours: float
    Adherence_to_Diet_Plan: Optional[float] = 0.5
    Dietary_Nutrient_Imbalance_Score: Optional[float] = 5.0


# ─────────────────────────────────────────
# POST /predict — main prediction endpoint
# ─────────────────────────────────────────
@app.post("/predict")
def predict(user: UserInput):
    try:
        # Convert pydantic model to dict
        user_dict = user.dict()

        # Rename fields to match training column names
        user_dict["Cholesterol_mg/dL"] = user_dict.pop("Cholesterol_mgdL")
        user_dict["Glucose_mg/dL"]     = user_dict.pop("Glucose_mgdL")

        # Apply same preprocessing as training
        X = preprocess_input(user_dict, scaler, label_encoders)

        # Run prediction
        pred_idx   = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        diet_type  = target_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(pred_proba[pred_idx]) * 100, 2)

        return {
            "diet_type": diet_type,
            "confidence": confidence,
            "all_probabilities": {
                cls: round(float(prob) * 100, 2)
                for cls, prob in zip(target_encoder.classes_, pred_proba)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# GET /health — health check
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Fuzzy Membership Layer + MLP (32,16) Stacked",
        "classes": feature_config["target_classes"]
    }