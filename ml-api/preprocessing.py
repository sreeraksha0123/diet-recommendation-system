"""
preprocessing.py
Shared preprocessing logic used in BOTH train_model.py AND main.py (FastAPI).
MUST be kept in sync with train_model.py at all times.

Model: PGS (Probabilistic Gradient Score) Layer stacked with MLP
  For each incoming patient record:
    1. Build raw numeric + encoded categorical feature vector
    2. Normalize numeric features with the fitted StandardScaler
    3. Compute K sigmoid risk scores via x @ W^T (W loaded from model/W_pgs.pkl)
    4. Compute 3 pairwise log-odds ratios
    5. Concatenate: 18 original + 3 risk + 3 log-odds = 24-dim input to MLP
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


def pgs_transform(X_scaled_in: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Applies PGS Layer: returns enriched matrix of shape (n, 24).

    Step B: z_k(x) = x . W[k]^T
            s_k(x) = sigmoid(z_k) — soft class-affinity score
    Step C: lo_{jk} = log(s_j / s_k) — pairwise log-odds
    Step D: concatenate [x_scaled || s_0 s_1 s_2 || lo_01 lo_02 lo_12]

    Args:
        X_scaled_in : shape (n, 18)
        W           : discriminant weight matrix, shape (K, 18)
    Returns:
        np.ndarray of shape (n, 24)
    """
    K    = W.shape[0]
    z    = X_scaled_in @ W.T                        # (n, K)
    risk = 1.0 / (1.0 + np.exp(-z))                # (n, K) sigmoid
    eps  = 1e-7
    lo   = []
    for j in range(K):
        for k in range(j + 1, K):
            lo.append(np.log((risk[:, j] + eps) / (risk[:, k] + eps)).reshape(-1, 1))
    return np.hstack([X_scaled_in, risk, np.hstack(lo)])


def preprocess_input(user_input: dict, scaler, label_encoders, W_pgs: np.ndarray) -> np.ndarray:
    """
    Applies same preprocessing as training to a single user input dict.
    Returns shape (1, 24) ready for model.predict().

    Args:
        user_input    : dict with keys matching feature names
        scaler        : fitted StandardScaler from model/scaler.pkl
        label_encoders: dict of fitted LabelEncoders from model/label_encoders.pkl
        W_pgs         : discriminant weight matrix from model/W_pgs.pkl

    Returns:
        np.ndarray of shape (1, 24)
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

    return pgs_transform(X_scaled, W_pgs)