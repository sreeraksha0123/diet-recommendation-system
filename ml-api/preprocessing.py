"""
preprocessing.py
Shared preprocessing logic used in BOTH train_model.py AND main.py (FastAPI).
MUST be kept in sync with train_model.py at all times.

Model: Fuzzy Membership Layer stacked with MLP
  For each incoming patient record, we:
    1. Build raw numeric + encoded categorical feature vector
    2. Normalize numeric features with the fitted StandardScaler
    3. Compute 18 fuzzy membership scores (6 features x 3 fuzzy sets each)
       using the same membership functions and thresholds as training
    4. Concatenate scaled features + fuzzy scores -> input to MLP
"""

import numpy as np

NUMERIC_FEATURES = [
    "Age", "Weight_kg", "Height_cm", "BMI",
    "Daily_Caloric_Intake", "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg", "Glucose_mg/dL",
    "Weekly_Exercise_Hours", "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score"
]

CATEGORICAL_FEATURES = [
    "Gender", "Disease_Type", "Severity",
    "Physical_Activity_Level", "Allergies",
    "Preferred_Cuisine", "Dietary_Restrictions"
]

DEFAULTS = {
    "Disease_Type": "None",
    "Allergies": "None",
    "Dietary_Restrictions": "None",
    "Adherence_to_Diet_Plan": 0.5,
    "Dietary_Nutrient_Imbalance_Score": 5.0,
}


def trimf(x, a, b, c):
    """Triangular membership function. Peaks at b, zero at a and c."""
    result = np.zeros_like(x, dtype=float)
    left  = (x > a) & (x <= b)
    right = (x > b) & (x < c)
    result[left]   = (x[left]  - a) / (b - a + 1e-9)
    result[right]  = (c - x[right]) / (c - b + 1e-9)
    result[x == b] = 1.0
    return np.clip(result, 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function. Flat top between b and c."""
    result = np.zeros_like(x, dtype=float)
    rise  = (x > a)  & (x < b)
    top   = (x >= b) & (x <= c)
    fall  = (x > c)  & (x < d)
    result[rise] = (x[rise] - a) / (b - a + 1e-9)
    result[top]  = 1.0
    result[fall] = (d - x[fall]) / (d - c + 1e-9)
    return np.clip(result, 0.0, 1.0)


FUZZY_RULES = [
    (3, "BMI", [
        ("low",  trapmf, (10,  10,  22,  25)),
        ("med",  trimf,  (22,  27,  32)),
        ("high", trapmf, (29,  32,  50,  50)),
    ]),
    (5, "Cholesterol", [
        ("low",  trapmf, (100, 100, 180, 200)),
        ("med",  trimf,  (180, 215, 245)),
        ("high", trapmf, (230, 250, 400, 400)),
    ]),
    (6, "BloodPressure", [
        ("low",  trapmf, (60,  60,  110, 120)),
        ("med",  trimf,  (110, 125, 140)),
        ("high", trapmf, (130, 145, 200, 200)),
    ]),
    (7, "Glucose", [
        ("low",  trapmf, (50,  50,  90,  100)),
        ("med",  trimf,  (90,  112, 130)),
        ("high", trapmf, (120, 135, 300, 300)),
    ]),
    (8, "Exercise", [
        ("low",  trapmf, (0,   0,   1.5, 2.5)),
        ("med",  trimf,  (1.5, 3.5, 5.5)),
        ("high", trapmf, (4.5, 6.0, 20,  20)),
    ]),
    (10, "NutrientImbalance", [
        ("low",  trapmf, (0,   0,   2,   4)),
        ("med",  trimf,  (2,   5,   8)),
        ("high", trapmf, (6,   8,   10,  10)),
    ]),
]


def fuzzy_membership_transform(X_raw, X_scaled_in):
    """Appends 18 fuzzy membership scores to the scaled feature matrix."""
    cols = []
    for feat_idx, _, fuzzy_sets in FUZZY_RULES:
        raw_col = X_raw[:, feat_idx]
        for _, fn, params in fuzzy_sets:
            cols.append(fn(raw_col, *params).reshape(-1, 1))
    return np.hstack([X_scaled_in] + cols)


def preprocess_input(user_input, scaler, label_encoders):
    """
    Applies same preprocessing as training to a single user input dict.
    Returns shape (1, 36) -- 18 original features + 18 fuzzy scores.
    """
    for key, default in DEFAULTS.items():
        if key not in user_input or user_input[key] is None:
            user_input[key] = default

    numeric_vals = [float(user_input[f]) for f in NUMERIC_FEATURES]

    cat_vals = []
    for col in CATEGORICAL_FEATURES:
        le  = label_encoders[col]
        val = str(user_input[col])
        encoded = int(le.transform([val])[0]) if val in le.classes_ else 0
        cat_vals.append(float(encoded))

    X_raw    = np.array([numeric_vals + cat_vals], dtype=float)
    X_scaled = X_raw.copy()
    X_scaled[:, :len(NUMERIC_FEATURES)] = scaler.transform(
        X_raw[:, :len(NUMERIC_FEATURES)]
    )

    return fuzzy_membership_transform(X_raw, X_scaled)