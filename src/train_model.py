from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# STEP 7 - TRAIN AND EVALUATE RANDOM FOREST MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "breakout_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


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

TARGET_COLUMN = "Breakout_5D"


def train_model():

    print("=" * 70)
    print("STEP 7 - RANDOM FOREST TRAINING AND EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load Feature Store
    # --------------------------------------------------------

    print("\nLoading Feature Store:")
    print(FEATURE_STORE_PATH)

    df = pd.read_csv(FEATURE_STORE_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Feature Store rows: {len(df):,}")
    print(f"Number of stocks: {df['Ticker'].nunique()}")

    # --------------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------------

    required_columns = (
        ["Date", "Ticker"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Use only labelled historical rows for model training
    # --------------------------------------------------------

    total_feature_store_rows = len(df)

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    print(
        f"Labelled rows used for "
        f"training/evaluation: {len(df):,}"
    )

    print(
        "Unlabelled latest rows retained "
        "for inference: "
        f"{total_feature_store_rows - len(df):,}"
    )
    # --------------------------------------------------------
    # 3. Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Prepare X and y
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    print(f"\nFeature columns: {len(FEATURE_COLUMNS)}")
    print("Target column:", TARGET_COLUMN)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------
    # 5. Chronological 70 / 15 / 15 split by trading date
    # --------------------------------------------------------

    unique_dates = sorted(df["Date"].unique())

    total_dates = len(unique_dates)

    train_date_end = int(total_dates * 0.70)
    validation_date_end = int(total_dates * 0.85)

    train_end_date = unique_dates[train_date_end - 1]
    validation_end_date = unique_dates[validation_date_end - 1]

    # Training = earliest 70%
    train_mask = df["Date"] <= train_end_date

    # Validation = next 15%
    validation_mask = (
        (df["Date"] > train_end_date)
        & (df["Date"] <= validation_end_date)
    )

    # Test = latest 15%
    test_mask = df["Date"] > validation_end_date

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 70)

    print(f"Total unique trading dates: {total_dates:,}")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")

    print("\nTraining period:")
    print(
        df.loc[train_mask, "Date"].min().date(),
        "to",
        df.loc[train_mask, "Date"].max().date()
    )

    print("\nValidation period:")
    print(
        df.loc[validation_mask, "Date"].min().date(),
        "to",
        df.loc[validation_mask, "Date"].max().date()
    )

    print("\nTest period:")
    print(
        df.loc[test_mask, "Date"].min().date(),
        "to",
        df.loc[test_mask, "Date"].max().date()
    )

    # --------------------------------------------------------
    # 6. Train Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Random Forest training completed.")

    # --------------------------------------------------------
    # 7. Save model
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # 8. Save exact feature column order
    # --------------------------------------------------------

    with open(FEATURE_COLUMNS_PATH, "w") as file:
        json.dump(
            FEATURE_COLUMNS,
            file,
            indent=4
        )

    print("\nFeature columns saved to:")
    print(FEATURE_COLUMNS_PATH)

    # --------------------------------------------------------
    # 9. Evaluate on unseen chronological test data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EVALUATION - UNSEEN TEST DATA")
    print("=" * 70)

    # Test data is not used during training.
    y_test_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        y_test_pred
    )

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    # --------------------------------------------------------
    # 10. Feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    for _, row in feature_importance.iterrows():
        print(
            f"{row['Feature']:<25} "
            f"{row['Importance']:.6f}"
        )

    # --------------------------------------------------------
    # 11. Check prototype targets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROTOTYPE TARGET CHECK")
    print("=" * 70)

    precision_target = 0.70
    recall_target = 0.60
    f1_target = 0.65

    print(
        f"Precision >= 70%: "
        f"{'PASS' if precision >= precision_target else 'NOT ACHIEVED'}"
    )

    print(
        f"Recall >= 60%:    "
        f"{'PASS' if recall >= recall_target else 'NOT ACHIEVED'}"
    )

    print(
        f"F1-score >= 65%:  "
        f"{'PASS' if f1 >= f1_target else 'NOT ACHIEVED'}"
    )

    # --------------------------------------------------------
    # 12. Save metrics
    # --------------------------------------------------------

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "confusion_matrix": confusion.tolist(),
        "test_rows": int(len(X_test)),
        "test_start_date": str(
            df.loc[test_mask, "Date"].min().date()
        ),
        "test_end_date": str(
            df.loc[test_mask, "Date"].max().date()
        ),
        "prototype_targets": {
            "precision": precision_target,
            "recall": recall_target,
            "f1_score": f1_target
        },
        "feature_importance": {
            row["Feature"]: round(
                float(row["Importance"]),
                6
            )
            for _, row in feature_importance.iterrows()
        }
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nMetrics saved to:")
    print(METRICS_PATH)

    # --------------------------------------------------------
    # 13. Final artifact validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ARTIFACT VALIDATION")
    print("=" * 70)

    print(
        "Model artifact exists:",
        MODEL_PATH.exists()
    )

    print(
        "Feature column file exists:",
        FEATURE_COLUMNS_PATH.exists()
    )

    print(
        "Metrics file exists:",
        METRICS_PATH.exists()
    )

    print("\n" + "=" * 70)
    print("STEP 7 - MODEL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_model()