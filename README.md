# Diet Recommendation ML API

FastAPI service that predicts a personalised diet type (Balanced / Low\_Carb / Low\_Sodium) from patient health metrics using a custom ML model.

## Project structure

```
ml-api/
├── data/
│   └── diet_recommendations_dataset.csv
├── model/                  # generated after training
│   ├── mlp_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   ├── gcllr_params.pkl
│   └── feature_config.json
├── train_model.py
├── preprocessing.py
├── main.py
└── requirements.txt
```

## Model

**Gaussian Class-Conditional Log-Likelihood Ratio (GCLLR) Layer stacked with MLP.**

For each patient the GCLLR layer computes, per diet class k:

```
log p(x | y=k) = Σ_j [ -½ ((x_j − μ_kj) / σ_kj)² − log(σ_kj) ]
```

where μ\_kj and σ\_kj are estimated from the training split only (no leakage). It also produces pairwise log-likelihood ratios between classes. These 6 scores are concatenated with the original 18 features to form a 24-dim input to an MLP(16, 8).

| Model | Accuracy |
|---|---|
| Plain MLP (baseline) | 84.00% |
| GCLLR + MLP (final) | 83.50% |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train (Phase 1)**
```bash
python train_model.py
```
Saves all model artifacts to `model/`.

**Serve (Phase 2)**
```bash
uvicorn main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

**Predict**
```bash
POST /predict
```
```json
{
  "Age": 45, "Gender": "Male", "Weight_kg": 85, "Height_cm": 170,
  "BMI": 29.4, "Disease_Type": "Diabetes", "Severity": "Moderate",
  "Physical_Activity_Level": "Sedentary", "Daily_Caloric_Intake": 2200,
  "Cholesterol_mgdL": 210, "Blood_Pressure_mmHg": 128,
  "Glucose_mgdL": 145, "Weekly_Exercise_Hours": 1.5
}
```
```json
{
  "diet_type": "Low_Carb",
  "confidence": 81.4,
  "all_probabilities": { "Balanced": 9.2, "Low_Carb": 81.4, "Low_Sodium": 9.4 }
}
```

**Health check**
```bash
GET /health
```
