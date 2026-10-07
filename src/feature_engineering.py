import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Only current and historical closing prices are used.
    """

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    Uses only current and historical OHLC data.
    """

    previous_close = close.shift(1)

    true_range_components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )

    true_range = true_range_components.max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=high.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=high.index,
    )

    smoothed_tr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * smoothed_plus_dm
        / smoothed_tr.replace(0, np.nan)
    )

    minus_di = (
        100
        * smoothed_minus_dm
        / smoothed_tr.replace(0, np.nan)
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / di_sum
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all agreed technical features.

    Features use only information available on or before
    the current date.

    The future Breakout_5D target is NOT created here.
    It will be created separately in the target-generation/
    feature-store stage.
    """

    required_columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    result["Date"] = pd.to_datetime(result["Date"])

    # Perform all calculations separately for each ticker
    # and in chronological order.
    result.sort_values(
        ["Ticker", "Date"],
        inplace=True,
    )

    # -----------------------------
    # EMA20
    # -----------------------------

    result["EMA20"] = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.ewm(
                span=20,
                adjust=False,
            ).mean()
        )
    )

    # -----------------------------
    # EMA50
    # -----------------------------

    result["EMA50"] = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.ewm(
                span=50,
                adjust=False,
            ).mean()
        )
    )

    # -----------------------------
    # RSI
    # -----------------------------

    result["RSI"] = (
        result.groupby("Ticker")["Close"]
        .transform(calculate_rsi)
    )

    # -----------------------------
    # MACD
    # -----------------------------

    ema12 = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.ewm(
                span=12,
                adjust=False,
            ).mean()
        )
    )

    ema26 = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.ewm(
                span=26,
                adjust=False,
            ).mean()
        )
    )

    result["MACD"] = ema12 - ema26

    # -----------------------------
    # ADX
    # -----------------------------
    #
    # Calculate separately for each ticker and explicitly
    # assign the result back using the original row index.
    # This avoids index-alignment problems with pandas
    # groupby/apply across different pandas versions.

    adx = pd.Series(
        np.nan,
        index=result.index,
        dtype=float,
    )

    for _, ticker_rows in result.groupby("Ticker"):
        ticker_adx = calculate_adx(
            ticker_rows["High"],
            ticker_rows["Low"],
            ticker_rows["Close"],
        )

        adx.loc[ticker_rows.index] = ticker_adx

    result["ADX"] = adx

    # -----------------------------
    # Bollinger Band Position
    # -----------------------------

    rolling_mean = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    rolling_std = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20,
            ).std()
        )
    )

    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    band_width = upper_band - lower_band

    result["BB_Position"] = (
        (result["Close"] - lower_band)
        / band_width.replace(0, np.nan)
    )

    # -----------------------------
    # Volume Ratio
    # -----------------------------

    average_volume_20 = (
        result.groupby("Ticker")["Volume"]
        .transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    result["Volume_Ratio"] = (
        result["Volume"]
        / average_volume_20.replace(0, np.nan)
    )

    # -----------------------------
    # Previous 20-Day High
    # -----------------------------
    #
    # IMPORTANT:
    # shift(1) means the current day's Close is NOT
    # included in the previous 20-day high.
    #
    # Therefore this feature uses only information
    # available before the current trading date.

    result["Previous_20D_High"] = (
        result.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=20,
                min_periods=20,
            ).max()
        )
    )

    result["Distance_20D_High"] = (
        (
            result["Previous_20D_High"]
            - result["Close"]
        )
        / result["Previous_20D_High"]
    )

    # -----------------------------
    # Returns
    # -----------------------------

    result["Return_1D"] = (
        result.groupby("Ticker")["Close"]
        .pct_change(
            periods=1,
            fill_method=None,
        )
    )

    result["Return_5D"] = (
        result.groupby("Ticker")["Close"]
        .pct_change(
            periods=5,
            fill_method=None,
        )
    )

    return result
