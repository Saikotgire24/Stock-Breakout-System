from pathlib import Path

import pandas as pd
import yfinance as yf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Current NIFTY 50 symbols
TICKERS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "BSE",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDIGO",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TCS",
    "TMPV",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
]

# Five years of daily historical data
PERIOD = "5y"
INTERVAL = "1d"


def download_stock_data(ticker: str) -> pd.DataFrame:
    """
    Download historical OHLCV data for one stock.
    """

    print(f"Downloading {ticker}...")

    # Yahoo Finance uses .NS suffix for NSE-listed stocks
    yahoo_ticker = f"{ticker}.NS"

    df = yf.download(
        yahoo_ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Handle yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{ticker}: Missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    # Convert index into Date column
    df.reset_index(inplace=True)

    # Keep only required assignment fields
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Add ticker identifier
    df["Ticker"] = ticker

    # Reorder columns according to the project requirement
    df = df[
        [
            "Date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    # Basic data cleaning
    df.dropna(inplace=True)
    df.drop_duplicates(
        subset=["Date", "Ticker"],
        inplace=True,
    )

    return df


def main():
    """
    Download data for all selected stocks and save
    the combined raw dataset.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_data = []

    for ticker in TICKERS:
        try:
            df = download_stock_data(ticker)

            # Save individual raw file
            individual_file = RAW_DATA_DIR / f"{ticker}.csv"
            df.to_csv(individual_file, index=False)

            all_data.append(df)

            print(
                f"{ticker}: {len(df):,} observations saved "
                f"to {individual_file}"
            )

        except Exception as exc:
            print(f"ERROR - {ticker}: {exc}")

    if not all_data:
        raise RuntimeError(
            "No stock data was downloaded successfully."
        )

    # Combine all stocks
    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined.sort_values(
        ["Date", "Ticker"],
        inplace=True,
    )

    combined.reset_index(
        drop=True,
        inplace=True,
    )

    combined_file = RAW_DATA_DIR / "stock_data.csv"
    combined.to_csv(
        combined_file,
        index=False,
    )

    # Convert Date for reporting
    combined["Date"] = pd.to_datetime(
        combined["Date"]
    )

    print("\n" + "=" * 60)
    print("STEP 2 - DATA ACQUISITION SUMMARY")
    print("=" * 60)

    print(f"Number of stocks: {combined['Ticker'].nunique()}")
    print(
        f"Stocks downloaded: "
        f"{sorted(combined['Ticker'].unique())}"
    )

    print(
        f"Date range: "
        f"{combined['Date'].min().date()} "
        f"to "
        f"{combined['Date'].max().date()}"
    )

    print(
        f"Total observations: "
        f"{len(combined):,}"
    )

    print(
        f"\nCombined raw data saved to:\n"
        f"{combined_file}"
    )

    print("\nFirst 5 rows:")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()