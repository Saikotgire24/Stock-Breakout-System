from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


# ============================================================
# STEP 8 - FEATURE SERVICE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_STORE_PATH = (
    PROJECT_ROOT / "data" / "feature_store.csv"
)


FEATURE_COLUMNS = [
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "ADX",
    "BB_Position",
    "Volume_Ratio",
    "Distance_20D_High",
    "Return_1D",
    "Return_5D",
]


# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Stock Breakout Feature Service",
    description="Reads latest engineered features from the Feature Store.",
    version="1.0.0",
)


# ------------------------------------------------------------
# Load Feature Store
# ------------------------------------------------------------

feature_store = pd.read_csv(
    FEATURE_STORE_PATH
)

feature_store["Date"] = pd.to_datetime(
    feature_store["Date"]
)

feature_store["Ticker"] = feature_store[
    "Ticker"
].str.upper()


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "feature-service",
        "feature_store": "loaded",
        "rows": len(feature_store),
        "stocks": int(
            feature_store["Ticker"].nunique()
        ),
    }


# ------------------------------------------------------------
# GET /features/{ticker}
# ------------------------------------------------------------

@app.get("/features/{ticker}")
def get_features(ticker: str):

    ticker = ticker.upper()

    ticker_data = feature_store[
        feature_store["Ticker"] == ticker
    ]

    if ticker_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found in Feature Store."
        )

    # Get the latest available record
    latest = ticker_data.sort_values(
        "Date"
    ).iloc[-1]

    response = {
        "ticker": ticker,
        "date": latest["Date"].date().isoformat(),
    }

    for column in FEATURE_COLUMNS:
        response[column] = float(
            latest[column]
        )

    return response