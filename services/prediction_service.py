from pathlib import Path

import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException


# ============================================================
# STEP 9 - PREDICTION SERVICE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "breakout_model.pkl"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "feature_columns.json"
)


# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Stock Breakout Prediction Service",
    description="Predicts stock breakout probability using the trained Random Forest model.",
    version="1.0.0",
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Load exact feature column order
# ------------------------------------------------------------

with open(FEATURE_COLUMNS_PATH, "r") as file:
    FEATURE_COLUMNS = json.load(file)


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "prediction-service",
        "model": "loaded",
        "features": len(FEATURE_COLUMNS),
    }


# ------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------

@app.post("/predict")
def predict(features: dict):

    # --------------------------------------------------------
    # Validate that all required features are present
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features,
            },
        )

    # --------------------------------------------------------
    # Create DataFrame using exact training feature order
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            [
                features[feature]
                for feature in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # Generate breakout probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]

    # --------------------------------------------------------
    # Generate predicted class
    # --------------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    return {
        "breakout_probability": round(
            float(probability),
            3
        ),
        "prediction": prediction,
    }