# ML-Based NIFTY 50 Stock Breakout Prediction and Alert System

**Course:** AIML ZG546 – Software Engineering for Machine Learning  
**Assignment:** Assignment I  
**Domain:** Financial Markets – Stock Market Analytics

## 1. Project Overview

This project implements an ML-based stock screening system for the **NIFTY 50** universe.

The system uses historical daily OHLCV market data and engineered technical indicators to estimate the probability that a stock will generate a bullish breakout within the next five trading days.

The application is intended as an **academic decision-support prototype**. It does not execute trades and should not be treated as financial advice.

### Breakout Definition

For a trading date `t`, a breakout is labelled as:

- `1` – Breakout
- `0` – No Breakout

`Breakout_5D = 1` when the maximum closing price during the next five trading days reaches or exceeds:

```text
1.02 × Previous_20D_High
```

where `Previous_20D_High` is calculated using historical observations available before the prediction date.

Future prices are used **only for target-label generation**, not as model input features.

---

## 2. Implemented System Architecture

```text
Historical Stock Market Data
          |
          v
    Data Ingestion
          |
          v
  Feature Engineering
          |
          v
     Feature Store
          |
       +--+------------------+
       |                     |
       v                     v
 Training Pipeline      Feature Service
       |                     |
       v                     v
 Random Forest         Prediction Service
                             |
                             v
                     Breakout Probability
                             |
                             v
                    Alert / Threshold Service
                             |
                             v
                         Final Result
```

### ML and Non-ML Components

| Component | Type | Purpose |
|---|---|---|
| Historical Stock Data | Non-ML | Provides OHLCV data |
| Data Ingestion | Non-ML | Downloads and prepares historical market data |
| Feature Engineering | Non-ML | Calculates reusable technical indicators |
| Feature Store | Non-ML | Stores prepared features for training and inference |
| Training Pipeline | ML | Trains the Random Forest classifier |
| Random Forest Model | ML | Produces breakout probabilities |
| Feature Service | Non-ML | Retrieves the latest feature record for a ticker |
| Prediction Service | ML | Performs model inference |
| Alert Service | Non-ML | Applies the configurable business alert threshold |

---

## 3. Architectural Patterns

### 3.1 Feature Store Pattern

A CSV-based Feature Store is implemented at:

```text
data/feature_store.csv
```

The same centralized feature representation is reused by:

1. Model training
2. Feature serving / inference

This reduces training-serving inconsistency and avoids duplicating feature logic.

The latest five observations for each stock are retained even though their `Breakout_5D` target is unavailable, because the required future five-day window does not yet exist. These rows remain available for inference.

### 3.2 Microservices Pattern

Three lightweight FastAPI services are implemented:

| Service | Port | Main Endpoints |
|---|---:|---|
| Feature Service | 8001 | `/health`, `/features/{ticker}` |
| Prediction Service | 8002 | `/health`, `/predict` |
| Alert Service | 8003 | `/health`, `/alert` |

The services communicate through HTTP/JSON.

The Alert Service separates ML inference from business decision logic. The model produces a breakout probability, while the Alert Service applies the configured business threshold.

---

## 4. Model Features

The Random Forest uses the following 10 engineered features:

```text
EMA20
EMA50
RSI
MACD
ADX
BB_Position
Volume_Ratio
Distance_20D_High
Return_1D
Return_5D
```

Target column:

```text
Breakout_5D
```

The feature-engineering logic uses only current and historical information. The future five-day window is used only for construction of the training target.

---

## 5. Project Structure

```text
SE4ML/
|
├── README.md
├── requirements.txt
|
├── data/
│   ├── raw/
│   │   └── stock_data.csv
│   └── feature_store.csv
|
├── artifacts/
│   ├── breakout_model.pkl
│   ├── feature_columns.json
│   └── metrics.json
|
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── create_feature_store.py
│   └── train_model.py
|
├── services/
│   ├── __init__.py
│   ├── feature_service.py
│   ├── prediction_service.py
│   └── alert_service.py
|
├── tests/
│   ├── test_basic.py
│   ├── test_end_to_end.py
│   └── test_end_to_end_batch.py
|
└── evidence/
    └── nifty50_batch_results.csv
```

---

## 6. Dataset and Feature Store

The latest implementation uses **50 NIFTY 50 stocks**.

Feature Store statistics:

```text
Feature Store rows                  : 60,277
Number of stocks                    : 50
Labelled rows for training/evaluation: 60,027
Latest unlabelled rows for inference : 250
```

The 250 unlabelled rows correspond to:

```text
50 stocks × latest 5 trading days
```

These records are intentionally retained for inference.

Historical market data is obtained using `yfinance` with NSE ticker symbols.

---

## 7. Environment Setup

Run commands from the project root.

### Windows / PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### macOS / Linux

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## 8. Data Preparation

### Step 1 – Load / Download Historical Market Data

```bash
python -m src.data_loader
```

This creates or updates:

```text
data/raw/stock_data.csv
```

### Step 2 – Create Feature Store

```bash
python -m src.create_feature_store
```

This generates:

```text
data/feature_store.csv
```

The Feature Store contains the engineered model features and `Breakout_5D` for historical labelled observations.

---

## 9. Model Training

Train the Random Forest model using:

```bash
python -m src.train_model
```

The training process:

- reads data from the Feature Store;
- excludes only rows where the target is unavailable;
- performs a chronological train/validation/test split;
- trains a `RandomForestClassifier`;
- evaluates the model on unseen chronological test data;
- saves the trained model and metadata.

Generated artifacts:

```text
artifacts/breakout_model.pkl
artifacts/feature_columns.json
artifacts/metrics.json
```

---

## 10. Chronological Split

The final NIFTY 50 model used:

| Dataset | Rows | Period |
|---|---:|---|
| Training | 41,887 | 2021-11-16 to 2025-04-22 |
| Validation | 9,050 | 2025-04-23 to 2026-01-09 |
| Test | 9,090 | 2026-01-12 to 2026-09-30 |

Total unique trading dates:

```text
1,210
```

The data is split chronologically rather than randomly to reduce time-series leakage.

---

## 11. Model Evaluation

### Actual Test Results

| Metric | Result |
|---|---:|
| Accuracy | 85.30% |
| Precision | 44.22% |
| Recall | 49.83% |
| F1-score | 46.86% |

Confusion matrix:

```text
[[7165  743]
 [ 593  589]]
```

Interpretation:

- True Negatives: 7,165
- False Positives: 743
- False Negatives: 593
- True Positives: 589

The dataset is imbalanced, so accuracy alone should not be used to judge breakout-detection quality. Precision, Recall and F1-score provide a more meaningful assessment of the positive breakout class.

### Prototype Target Check

Initial prototype targets were:

```text
Precision >= 70%
Recall    >= 60%
F1-score  >= 65%
```

Actual outcome:

| Metric | Target | Actual | Result |
|---|---:|---:|---|
| Precision | 70% | 44.22% | Not achieved |
| Recall | 60% | 49.83% | Not achieved |
| F1-score | 65% | 46.86% | Not achieved |

The model therefore demonstrates a working end-to-end ML system but requires further optimization before production use.

### Feature Importance

| Feature | Importance |
|---|---:|
| Distance_20D_High | 0.220702 |
| BB_Position | 0.157902 |
| RSI | 0.125209 |
| MACD | 0.081312 |
| Return_5D | 0.079409 |
| Return_1D | 0.073169 |
| ADX | 0.069356 |
| Volume_Ratio | 0.065416 |
| EMA50 | 0.064324 |
| EMA20 | 0.063200 |

The three most influential features are:

1. Distance from the previous 20-day high
2. Bollinger Band position
3. RSI

Feature importance indicates how strongly the Random Forest uses each feature; it should not be interpreted as proof of causality.

---

## 12. Automated Testing

Run:

```bash
python -m pytest tests/test_basic.py -v
```

Current result:

```text
6 passed
```

The automated tests cover:

1. Feature Store contains the required columns
2. Model artifact exists
3. Prediction probability is between 0 and 1
4. Breakout target uses the correct future five-day window
5. Alert threshold logic works correctly
6. FastAPI health endpoints return success through `TestClient`

Unit/component tests do not require the three API servers to be started manually.

---

## 13. Start the FastAPI Services

For end-to-end execution, start each service in a separate terminal.

### Terminal 1 – Feature Service

```bash
uvicorn services.feature_service:app --host 127.0.0.1 --port 8001
```

Swagger UI:

```text
http://127.0.0.1:8001/docs
```

### Terminal 2 – Prediction Service

```bash
uvicorn services.prediction_service:app --host 127.0.0.1 --port 8002
```

Swagger UI:

```text
http://127.0.0.1:8002/docs
```

### Terminal 3 – Alert Service

```bash
uvicorn services.alert_service:app --host 127.0.0.1 --port 8003
```

Swagger UI:

```text
http://127.0.0.1:8003/docs
```

---

## 14. API Flow

### Feature Service

Example:

```http
GET http://127.0.0.1:8001/features/RELIANCE
```

The Feature Service returns the latest available engineered feature record for the requested ticker.

### Prediction Service

```http
POST http://127.0.0.1:8002/predict
Content-Type: application/json
```

Example output:

```json
{
  "breakout_probability": 0.64,
  "prediction": 1
}
```

The ML classification threshold and the business alert threshold are intentionally separate.

### Alert Service

```http
POST http://127.0.0.1:8003/alert
Content-Type: application/json
```

Example:

```json
{
  "breakout_probability": 0.80
}
```

Response:

```json
{
  "breakout_probability": 0.80,
  "threshold": 0.75,
  "alert": true,
  "decision": "Potential Breakout"
}
```

The current business alert threshold is:

```text
0.75
```

---

## 15. Single-Stock End-to-End Demonstration

With all three services running:

```bash
python tests/test_end_to_end.py RELIANCE
```

The flow is:

```text
Ticker
  |
  v
Feature Store
  |
  v
Feature Service
  |
  v
Prediction Service
  |
  v
Breakout Probability
  |
  v
Alert Service
  |
  v
Final Result
```

---

## 16. NIFTY 50 Batch Demonstration

The system also supports an end-to-end batch test for the complete 50-stock universe.

Run:

```bash
python tests/test_end_to_end_batch.py
```

The batch test sends each stock through:

```text
Feature Service
      |
      v
Prediction Service
      |
      v
Alert Service
```

and ranks stocks by breakout probability.

### Latest Batch Result

The latest recorded batch evidence contains:

```text
Stocks requested       : 50
Successfully processed : 50
Failed                 : 0
Potential breakouts    : 1
```

Processing performance:

```text
Total processing time  : ~4.72 seconds
Average time per stock : ~0.094 seconds
95th percentile        : ~0.118 seconds
```

Top five stocks in the recorded batch:

| Rank | Ticker | Breakout Probability | Decision |
|---:|---|---:|---|
| 1 | KOTAKBANK | 0.905 | Potential Breakout |
| 2 | INDIGO | 0.640 | No Alert |
| 3 | AXISBANK | 0.630 | No Alert |
| 4 | BSE | 0.560 | No Alert |
| 5 | ITC | 0.500 | No Alert |

The batch result is stored at:

```text
evidence/nifty50_batch_results.csv
```

This batch test demonstrates the business use case of screening and ranking an entire stock universe rather than processing stocks manually one at a time.

---

## 17. Evidence for Assignment Report

Recommended evidence/screenshots:

1. Historical NIFTY 50 data loaded successfully
2. Feature Store showing engineered feature columns
3. Feature Store summary showing 50 stocks and 250 latest unlabelled inference rows
4. Random Forest training completed
5. Model evaluation metrics and confusion matrix
6. Feature importance output
7. `6 passed` automated test result
8. Feature Service Swagger/API response
9. Prediction Service Swagger/API response
10. Alert Service showing `Potential Breakout` / `No Alert`
11. NIFTY 50 batch processing output
12. Ranked batch candidates and summary

All screenshots should be captured from the actual running implementation.

---

## 18. Limitations and Assumptions

- This is an academic prototype, not a production trading system.
- The system performs screening and prediction only; it does not execute trades.
- Predictive performance did not achieve the initial prototype Precision, Recall and F1 targets.
- The stock universe is limited to the NIFTY 50 for the implemented demonstration.
- The alert threshold is currently configured at `0.75`.
- Historical technical indicators alone may not capture all factors affecting market breakouts.
- No live brokerage integration, portfolio management or automated order execution is included.
- NIFTY 50 membership changes over time; the configured ticker list represents the universe used for this implementation.
- Further work could include model tuning, class-imbalance handling, additional features and threshold optimization.

---

## 19. Assignment Scope

The implementation intentionally focuses on the AIML ZG546 Assignment I requirements:

- ML-enabled system architecture
- Feature Store Pattern
- Microservices Pattern
- Working Random Forest model
- Chronological model evaluation
- Working Feature Service
- Working Prediction Service
- Working Alert Service
- Automated testing
- Single-stock end-to-end execution
- NIFTY 50 batch execution
- Evidence generation

Unnecessary production infrastructure such as Kubernetes, Docker, Kafka, Redis, MLflow, a frontend, brokerage integration or automated trading has intentionally not been added.
