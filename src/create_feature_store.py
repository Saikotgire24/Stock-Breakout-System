from pathlib import Path

import pandas as pd

from src.feature_engineering import create_features


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stock_data.csv"
)

FEATURE_STORE_FILE = (
    PROJECT_ROOT
    / "data"
    / "feature_store.csv"
)


# ---------------------------------------------------------
# Final ML feature columns
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Step 4 - Breakout target generation
# ---------------------------------------------------------

def create_breakout_target(
    df: pd.DataFrame,
    threshold: float = 1.02,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Create the leakage-safe Breakout_5D target.

    Breakout_5D = 1 when the maximum Close over the
    NEXT 5 trading days (t+1 ... t+5) is at least:

        1.02 * Previous_20D_High

    Otherwise:

        Breakout_5D = 0

    Future prices are used ONLY to create the target.
    They are never used as model input features.
    """

    result = df.copy()

    result.sort_values(
        ["Ticker", "Date"],
        inplace=True,
    )

    # -----------------------------------------------------
    # Correct maximum Close over NEXT horizon trading days
    # -----------------------------------------------------

    def future_window_max(close: pd.Series) -> pd.Series:
        """
        For each row t, calculate:

        max(Close[t+1], ..., Close[t+horizon])

        All horizon future rows must exist.
        """

        future_values = pd.concat(
            [
                close.shift(-step)
                for step in range(1, horizon + 1)
            ],
            axis=1,
        )

        future_max = future_values.max(axis=1)

        # Last 'horizon' rows do not have enough future data.
        complete_window = (
            future_values.notna().sum(axis=1)
            == horizon
        )

        future_max = future_max.where(
            complete_window
        )

        return future_max

    result["Future_5D_Max_Close"] = (
        result.groupby("Ticker")["Close"]
        .transform(future_window_max)
    )

    # -----------------------------------------------------
    # Breakout threshold
    # -----------------------------------------------------

    result["Breakout_Threshold"] = (
        result["Previous_20D_High"]
        * threshold
    )

    # -----------------------------------------------------
    # Breakout target
    # -----------------------------------------------------

    result["Breakout_5D"] = (
        result["Future_5D_Max_Close"]
        >= result["Breakout_Threshold"]
    ).astype("Int64")

    # Target cannot be calculated when:
    # 1. Previous 20-day high is unavailable
    # 2. Full next-5-day window is unavailable

    invalid_target_rows = (
        result["Previous_20D_High"].isna()
        | result["Future_5D_Max_Close"].isna()
    )

    result.loc[
        invalid_target_rows,
        "Breakout_5D",
    ] = pd.NA

    return result


# ---------------------------------------------------------
# Prepare complete dataset
# ---------------------------------------------------------

def prepare_data_with_target(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate technical features and the Breakout_5D target.
    """

    # Step 3 - Feature engineering
    result = create_features(raw_df)

    # Step 4 - Target generation
    result = create_breakout_target(result)

    return result


# ---------------------------------------------------------
# Step 5 - Create Feature Store
# ---------------------------------------------------------

def create_feature_store() -> pd.DataFrame:
    """
    Create the final centralized Feature Store.

    The Feature Store contains:
        - Date
        - Ticker
        - agreed ML features
        - Breakout_5D target

    Rows with insufficient feature or target information
    are removed before saving.
    """

    print("=" * 70)
    print("STEP 5 - FEATURE STORE CREATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load raw data
    # -----------------------------------------------------

    print(f"\nLoading raw data:")
    print(RAW_DATA_FILE)

    raw_df = pd.read_csv(RAW_DATA_FILE)

    print(f"Raw observations: {len(raw_df):,}")

    # -----------------------------------------------------
    # Generate features + target
    # -----------------------------------------------------

    prepared_df = prepare_data_with_target(raw_df)

    print(
        f"Rows after feature/target generation: "
        f"{len(prepared_df):,}"
    )

    # -----------------------------------------------------
    # Columns required for final Feature Store
    # -----------------------------------------------------

    required_columns = [
        "Date",
        "Ticker",
        "Close",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in prepared_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Keep only required Feature Store columns
    # -----------------------------------------------------

    feature_store = prepared_df[
        required_columns
    ].copy()

    # -----------------------------------------------------
    # Remove rows without sufficient history
    # -----------------------------------------------------
    #
    # This removes rows where any required feature or
    # Breakout_5D target cannot be determined.
    #
    # Raw data remains untouched.

    before_drop = len(feature_store)

    # -----------------------------------------------------
    # Remove only rows with incomplete INPUT FEATURES
    # -----------------------------------------------------
    #
    # Do NOT remove rows merely because Breakout_5D is
    # unavailable.
    #
    # The latest 5 trading days naturally do not have a
    # future 5-day label yet, but their features are still
    # useful for inference.

    before_drop = len(feature_store)

    feature_store.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True,
    )

    rows_removed = (
            before_drop
            - len(feature_store)
    )

    # -----------------------------------------------------
    # Keep target nullable
    # -----------------------------------------------------

    feature_store[TARGET_COLUMN] = (
        feature_store[TARGET_COLUMN]
        .astype("Int64")
    )

    # -----------------------------------------------------
    # Validate labelled target values
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Validate labelled target values
    # -----------------------------------------------------

    labelled_targets = (
        feature_store.loc[
            feature_store[TARGET_COLUMN].notna(),
            TARGET_COLUMN
        ]
        .astype("int64")
    )

    unique_targets = sorted(
        labelled_targets.unique().tolist()
    )

    print(
        "Unique labelled target values:",
        unique_targets
    )

    if set(unique_targets) != {0, 1}:
        raise ValueError(
            "Labelled Breakout_5D rows must contain "
            "both 0 and 1. "
            f"Found: {unique_targets}"
        )

    if unique_targets != [0, 1]:
        raise ValueError(
            "Labelled Breakout_5D rows must "
            "contain both 0 and 1. "
            f"Found: {unique_targets}"
        )

    # -----------------------------------------------------
    # Final sorting
    # -----------------------------------------------------

    feature_store.sort_values(
        ["Ticker", "Date"],
        inplace=True,
    )

    feature_store.reset_index(
        drop=True,
        inplace=True,
    )

    # -----------------------------------------------------
    # Save Feature Store
    # -----------------------------------------------------

    FEATURE_STORE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_store.to_csv(
        FEATURE_STORE_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Final validation output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE STORE SUMMARY")
    print("=" * 70)

    print(
        f"Rows before removing incomplete data: "
        f"{before_drop:,}"
    )

    print(
        f"Rows removed due to insufficient history: "
        f"{rows_removed:,}"
    )

    print(
        f"Final Feature Store rows: "
        f"{len(feature_store):,}"
    )

    print(
        f"Number of stocks: "
        f"{feature_store['Ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{feature_store['Date'].min()} "
        f"to "
        f"{feature_store['Date'].max()}"
    )

    print("\nFeature Store columns:")

    for column in feature_store.columns:
        print(f"  - {column}")

    print("\nLabelled target distribution:")

    print(
        feature_store[TARGET_COLUMN]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLabelled target distribution (%):")

    print(
        (
                feature_store[TARGET_COLUMN]
                .dropna()
                .astype(int)
                .value_counts(normalize=True)
                .sort_index()
                * 100
        )
        .round(2)
        .to_string()
    )

    print(
        "\nRows without target "
        "(kept for inference):",
        int(
            feature_store[
                TARGET_COLUMN
            ].isna().sum()
        ),
    )

    print("\nTarget distribution (%):")

    print(
        (
            feature_store[TARGET_COLUMN]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )
        .round(2)
        .to_string()
    )

    print("\nMissing values in final Feature Store:")

    print(
        feature_store[
            FEATURE_COLUMNS + [TARGET_COLUMN]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print("\nFirst 5 Feature Store rows:")

    print(
        feature_store.head(5)
        .to_string(index=False)
    )

    print("\nFeature Store saved to:")

    print(FEATURE_STORE_FILE)

    print("\n" + "=" * 70)
    print("STEP 5 - FEATURE STORE CREATION COMPLETE")
    print("=" * 70)

    return feature_store


if __name__ == "__main__":
    create_feature_store()