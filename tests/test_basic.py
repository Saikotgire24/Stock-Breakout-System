from pathlib import Path
import json
from urllib.request import urlopen
from services.alert_service import create_alert
from fastapi.testclient import TestClient

from services.feature_service import app as feature_app
from services.prediction_service import app as prediction_app
from services.alert_service import app as alert_app
import joblib
import pandas as pd

from src.create_feature_store import (
    create_breakout_target
)
# ============================================================
# STEP 12 - BASIC TESTING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_STORE_PATH = (
    PROJECT_ROOT / "data" / "feature_store.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "breakout_model.pkl"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT / "artifacts" / "feature_columns.json"
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
# Test 1 - Feature engineering / Feature Store columns
# ------------------------------------------------------------
def test_breakout_target_uses_next_five_days():

    dates = pd.date_range(
        start="2026-01-01",
        periods=12,
        freq="B",
    )

    # Important construction:
    #
    # Row 4:
    # a large value exists in the PAST,
    # but the next 5 days do NOT breakout.
    #
    # Row 5:
    # the next 5 days contain 103,
    # therefore it SHOULD be breakout.

    df = pd.DataFrame({
        "Date": dates,
        "Ticker": ["TEST"] * 12,
        "Close": [
            100,
            150,   # old/past spike
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            103,   # future breakout
            100,
        ],
        "Previous_20D_High": [
            100
        ] * 12,
    })

    result = create_breakout_target(
        df,
        threshold=1.02,
        horizon=5,
    )

    # --------------------------------------------------
    # Row 4:
    #
    # Next 5 closes = rows 5..9
    # [100, 100, 100, 100, 100]
    #
    # No breakout.
    # The earlier 150 MUST NOT affect this target.
    # --------------------------------------------------

    assert (
        result.loc[
            result.index[4],
            "Breakout_5D"
        ]
        == 0
    )

    # --------------------------------------------------
    # Row 5:
    #
    # Next 5 closes = rows 6..10
    # includes 103
    #
    # 103 >= 1.02 * 100
    # therefore breakout = 1
    # --------------------------------------------------

    assert (
        result.loc[
            result.index[5],
            "Breakout_5D"
        ]
        == 1
    )

    # --------------------------------------------------
    # Final 5 rows cannot have a complete future window
    # --------------------------------------------------

    assert (
        result.tail(5)[
            "Breakout_5D"
        ].isna().all()
    )

def test_feature_store_has_required_columns():

    df = pd.read_csv(
        FEATURE_STORE_PATH
    )

    required_columns = [
        "Date",
        "Ticker",
    ] + FEATURE_COLUMNS + [
        "Breakout_5D"
    ]

    for column in required_columns:
        assert column in df.columns

    print(
        "\nTest 1 PASS: "
        "Feature Store contains all required columns."
    )


# ------------------------------------------------------------
# Test 2 - Model artifact exists
# ------------------------------------------------------------

def test_model_artifact_exists():

    assert MODEL_PATH.exists()

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None

    print(
        "\nTest 2 PASS: "
        "Random Forest model artifact exists."
    )


# ------------------------------------------------------------
# Test 3 - Prediction probability between 0 and 1
# ------------------------------------------------------------

def test_prediction_probability_range():

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        FEATURE_COLUMNS_PATH,
        "r"
    ) as file:
        feature_columns = json.load(file)

    df = pd.read_csv(
        FEATURE_STORE_PATH
    )

    sample = df[
        feature_columns
    ].iloc[[0]]

    probability = model.predict_proba(
        sample
    )[0][1]

    assert 0.0 <= probability <= 1.0

    print(
        f"\nTest 3 PASS: "
        f"Prediction probability = {probability:.4f}"
    )


# ------------------------------------------------------------
# Test 4 - Alert threshold logic
# ------------------------------------------------------------

def test_alert_threshold_logic():

    low_result = create_alert({
        "breakout_probability": 0.50
    })

    assert low_result["alert"] is False
    assert (
        low_result["decision"]
        == "No Alert"
    )

    high_result = create_alert({
        "breakout_probability": 0.80
    })

    assert high_result["alert"] is True
    assert (
        high_result["decision"]
        == "Potential Breakout"
    )

    print(
        "\nTest 4 PASS: "
        "Alert threshold logic works correctly."
    )
# ------------------------------------------------------------
# Test 5 - API health test
# ------------------------------------------------------------

def test_api_health_endpoints():

    feature_client = TestClient(feature_app)
    prediction_client = TestClient(prediction_app)
    alert_client = TestClient(alert_app)

    # Feature Service
    response = feature_client.get("/health")

    assert response.status_code == 200

    feature_result = response.json()

    assert (
        feature_result["status"]
        == "healthy"
    )

    # Prediction Service
    response = prediction_client.get("/health")

    assert response.status_code == 200

    prediction_result = response.json()

    assert (
        prediction_result["status"]
        == "healthy"
    )

    # Alert Service
    response = alert_client.get("/health")

    assert response.status_code == 200

    alert_result = response.json()

    assert (
        alert_result["status"]
        == "healthy"
    )