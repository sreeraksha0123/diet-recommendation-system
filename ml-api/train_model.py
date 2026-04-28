"""
Phase 1: Diet Recommendation ML Model Training
Architecture: Probabilistic Gradient Score (PGS) Layer stacked with MLP

Novel Mathematical Model — PGS Layer:
  Computes K class-discriminant risk scores via sigmoid(X @ W^T) where
  W[k] = class_centroid_k - global_centroid (analytic, no backprop).
  Appends 3 pairwise log-odds ratios. Stacks 6 new features with
  the original 18 scaled features -> 24-dim input to MLP.

Result:
  Plain MLP (16,8):          84.00%
  PGS + MLP (16,8) stacked:  86.50%   <- final model
"""

import pandas as pd
import numpy as np
import os, pickle, json, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# ── 1. LOAD ──
print("=" * 60)
print("  PHASE 1: Diet Recommendation ML Training")
print("  Model: PGS Layer (custom) stacked with MLP")
print("=" * 60)

df = pd.read_csv("data/diet_recommendations_dataset.csv")
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Target classes: {df['Diet_Recommendation'].unique().tolist()}")

# ── 2. PREPROCESS ──
print("\n[2] Preprocessing...")
df.drop(columns=["Patient_ID"], inplace=True)
for col in ["Disease_Type", "Dietary_Restrictions", "Allergies"]:
    df[col] = df[col].fillna("None")

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
TARGET = "Diet_Recommendation"

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"    Encoded '{col}': {list(le.classes_)}")

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET])
n_classes = len(target_encoder.classes_)
print(f"\n    Target classes: {list(target_encoder.classes_)}")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].values.astype(float)
print(f"    Feature matrix shape: {X.shape}")

# ── 3. SCALE ──
print("\n[3] Normalizing numeric features...")
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, :len(NUMERIC_FEATURES)] = scaler.fit_transform(X[:, :len(NUMERIC_FEATURES)])

# ── 4. SPLIT (before PGS fit -- no leakage) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[4] Train/Test split -- Train: {X_train.shape}, Test: {X_test.shape}")

# ── 5. PGS LAYER ──────────────────────────────────────────────────────────────
# Probabilistic Gradient Score Layer  (custom mathematical model)
#
# FORMULATION
# -----------
# Given training data X_train in R^{nxd} with y_train in {0,...,K-1}:
#
#   Step A -- Discriminant Weight Matrix  W in R^{Kxd}
#     mu_k = mean of X_train[y_train == k]   (class centroid)
#     mu   = mean of X_train                 (global centroid)
#     W[k] = mu_k - mu                       (discriminant direction)
#
#     W encodes: direction in feature space along which class k
#     most strongly deviates from the population average.
#     Purely analytic -- fitted once on training data, fixed at inference.
#
#   Step B -- Sigmoid Risk Scores  s in R^{nxK}
#     z_k(x) = x . W[k]^T                   (dot product projection)
#     s_k(x) = 1 / (1 + exp(-z_k(x)))       (sigmoid squash to (0,1))
#
#     s_k(x) is a soft score of how "class-k-like" patient x is.
#
#   Step C -- Pairwise Log-Odds Ratios  lo in R^{n x C(K,2)}
#     lo_{jk}(x) = log( s_j(x) / s_k(x) )  for all j < k
#     For K=3: pairs (0,1),(0,2),(1,2) -> 3 terms
#
#     Log-odds provide scale-free contrast between class affinities.
#
#   Step D -- Concatenation (the stacking)
#     x_enriched = [ x_scaled || s_0 s_1 s_2 || lo_01 lo_02 lo_12 ]
#                in R^{ d + K + C(K,2) } = R^{18+3+3} = R^24
#
# NO label information is used at inference. W is a fixed matrix loaded
# from model/W_pgs.pkl, computed analytically from training centroids.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Fitting PGS Layer (Probabilistic Gradient Score)...")

# Step A
class_centroids = np.array([
    X_train[y_train == k].mean(axis=0) for k in range(n_classes)
])                                          # (K, d)
global_centroid = X_train.mean(axis=0)     # (d,)
W_pgs = class_centroids - global_centroid  # (K, d)

print(f"    W_pgs shape:         {W_pgs.shape}  (K classes x d features)")
print(f"    Discriminant norms:  {[f'{np.linalg.norm(W_pgs[k]):.4f}' for k in range(n_classes)]}")


def pgs_transform(X_scaled_in, W):
    """
    Applies PGS Layer: returns enriched matrix of shape (n, d+K+C(K,2)).
    For K=3, d=18: output shape is (n, 24).

    Args:
        X_scaled_in : shape (n, d)
        W           : discriminant weight matrix, shape (K, d)
    Returns:
        np.ndarray of shape (n, d + K + K*(K-1)//2)
    """
    K    = W.shape[0]
    # Step B
    z    = X_scaled_in @ W.T               # (n, K)
    risk = 1.0 / (1.0 + np.exp(-z))       # (n, K)
    # Step C
    eps  = 1e-7
    lo   = []
    for j in range(K):
        for k in range(j + 1, K):
            lo.append(np.log((risk[:, j] + eps) / (risk[:, k] + eps)).reshape(-1, 1))
    # Step D
    return np.hstack([X_scaled_in, risk, np.hstack(lo)])


X_train_pgs = pgs_transform(X_train, W_pgs)   # (800, 24)
X_test_pgs  = pgs_transform(X_test,  W_pgs)   # (200, 24)

print(f"\n    Original features:      {X_train.shape[1]}")
print(f"    PGS risk scores:        {n_classes}  (sigmoid projections, one per class)")
print(f"    Log-odds terms:         3  (pairwise contrasts: 0v1, 0v2, 1v2)")
print(f"    Enriched shape:         {X_train_pgs.shape}")

s_sample = pgs_transform(X_test[:1], W_pgs)
d = X_test.shape[1]
print(f"\n    Sample PGS (test[0]):  risk={[round(float(s_sample[0,d+k]),4) for k in range(n_classes)]}  "
      f"log-odds={[round(float(s_sample[0,d+n_classes+i]),4) for i in range(3)]}")

# ── 6. TRAIN MODELS ──
print("\n[6] Training models...")

mlp_plain = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
    alpha=0.01, max_iter=50, random_state=42
)
mlp_plain.fit(X_train, y_train)

mlp_pgs = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
    alpha=0.01, max_iter=30, random_state=42
)
mlp_pgs.fit(X_train_pgs, y_train)

# ── 7. COMPARISON ──
acc_plain = accuracy_score(y_test, mlp_plain.predict(X_test))
acc_pgs   = accuracy_score(y_test, mlp_pgs.predict(X_test_pgs))

print("\n[7] Model Comparison:")
print("-" * 60)
print(f"    Plain MLP (16,8):                    {acc_plain * 100:.2f}%")
print(f"    PGS Layer + MLP (16,8) [Stacked]:    {acc_pgs   * 100:.2f}%")

# ── 8. FINAL EVAL ──
print("\n[8] Final Model Evaluation (PGS + MLP):")
print("=" * 60)

y_pred = mlp_pgs.predict(X_test_pgs)
acc    = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 9. SAVE ──
print("\n[9] Saving artifacts...")
os.makedirs("model", exist_ok=True)

with open("model/mlp_model.pkl",      "wb") as f: pickle.dump(mlp_pgs,       f)
with open("model/scaler.pkl",         "wb") as f: pickle.dump(scaler,         f)
with open("model/label_encoders.pkl", "wb") as f: pickle.dump(label_encoders, f)
with open("model/target_encoder.pkl", "wb") as f: pickle.dump(target_encoder, f)
with open("model/W_pgs.pkl",          "wb") as f: pickle.dump(W_pgs,          f)

feature_config = {
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features_order":   NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "n_classes":            n_classes,
    "pgs_features_added":   n_classes + (n_classes * (n_classes - 1) // 2),
    "enriched_dim":         int(X_train_pgs.shape[1]),
    "target_classes":       list(target_encoder.classes_),
    "plain_mlp_accuracy":   round(float(acc_plain), 4),
    "pgs_mlp_accuracy":     round(float(acc_pgs),   4),
    "model_used":           "pgs_stacked_mlp"
}
with open("model/feature_config.json", "w") as f:
    json.dump(feature_config, f, indent=2)

print("    model/mlp_model.pkl       <- PGS + MLP stacked model")
print("    model/W_pgs.pkl           <- discriminant weight matrix")
print("    model/scaler.pkl")
print("    model/label_encoders.pkl")
print("    model/target_encoder.pkl")
print("    model/feature_config.json")

print(f"\n{'='*60}")
print(f"  Phase 1 COMPLETE")
print(f"  Plain MLP:          {acc_plain*100:.2f}%")
print(f"  PGS + MLP:          {acc_pgs*100:.2f}%  <- final model")
print(f"{'='*60}")