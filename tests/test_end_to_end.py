from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import csv
import json
import time


# ============================================================
# NIFTY 50 BATCH END-TO-END DEMONSTRATION
# ============================================================

FEATURE_SERVICE_URL = "http://127.0.0.1:8001"
PREDICTION_SERVICE_URL = "http://127.0.0.1:8002"
ALERT_SERVICE_URL = "http://127.0.0.1:8003"


# ------------------------------------------------------------
# Current NIFTY 50 constituents
# ------------------------------------------------------------

NIFTY_50 = [
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


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

EVIDENCE_DIR = (
    PROJECT_ROOT / "evidence"
)

RESULTS_FILE = (
    EVIDENCE_DIR
    / "nifty50_batch_results.csv"
)


# ============================================================
# HTTP HELPERS
# ============================================================

def get_json(url):
    """
    Send GET request and return:
        HTTP status, decoded JSON response.
    """

    try:

        with urlopen(
            url,
            timeout=15
        ) as response:

            return (
                response.status,
                json.loads(
                    response
                    .read()
                    .decode("utf-8")
                ),
            )

    except HTTPError as error:

        error_body = (
            error.read()
            .decode("utf-8")
        )

        raise RuntimeError(
            f"GET {url} failed "
            f"(HTTP {error.code}) "
            f"{error_body}"
        )

    except URLError as error:

        raise RuntimeError(
            f"Could not connect to "
            f"{url}: {error.reason}"
        )


def post_json(url, payload):
    """
    Send POST request with JSON payload and
    return HTTP status and decoded JSON response.
    """

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = Request(
        url,
        data=data,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST",
    )

    try:

        with urlopen(
            request,
            timeout=15
        ) as response:

            return (
                response.status,
                json.loads(
                    response
                    .read()
                    .decode("utf-8")
                ),
            )

    except HTTPError as error:

        error_body = (
            error.read()
            .decode("utf-8")
        )

        raise RuntimeError(
            f"POST {url} failed "
            f"(HTTP {error.code}) "
            f"{error_body}"
        )

    except URLError as error:

        raise RuntimeError(
            f"Could not connect to "
            f"{url}: {error.reason}"
        )


# ============================================================
# SERVICE HEALTH CHECK
# ============================================================

def verify_services():
    """
    Confirm all three FastAPI services are running
    before starting the 50-stock batch test.
    """

    services = {
        "Feature Service":
            f"{FEATURE_SERVICE_URL}/health",

        "Prediction Service":
            f"{PREDICTION_SERVICE_URL}/health",

        "Alert Service":
            f"{ALERT_SERVICE_URL}/health",
    }

    print("\nChecking services...")

    for name, url in services.items():

        status, response = get_json(
            url
        )

        print(
            f"{name:<22} "
            f"HTTP {status} "
            f"- {response.get('status')}"
        )


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(ticker):
    """
    Run one stock through:

    Feature Service
        ->
    Prediction Service
        ->
    Alert Service
    """

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 1. FEATURE SERVICE
    # --------------------------------------------------------

    feature_status, feature_data = get_json(
        f"{FEATURE_SERVICE_URL}"
        f"/features/{ticker}"
    )

    if feature_status != 200:

        raise RuntimeError(
            "Feature Service failed."
        )

    # --------------------------------------------------------
    # Build prediction input
    # --------------------------------------------------------

    prediction_payload = {
        feature:
            feature_data[feature]

        for feature
        in FEATURE_COLUMNS
    }

    # --------------------------------------------------------
    # 2. PREDICTION SERVICE
    # --------------------------------------------------------

    (
        prediction_status,
        prediction_data,
    ) = post_json(
        f"{PREDICTION_SERVICE_URL}"
        f"/predict",
        prediction_payload,
    )

    if prediction_status != 200:

        raise RuntimeError(
            "Prediction Service failed."
        )

    probability = (
        prediction_data[
            "breakout_probability"
        ]
    )

    model_prediction = (
        prediction_data[
            "prediction"
        ]
    )

    # --------------------------------------------------------
    # 3. ALERT SERVICE
    # --------------------------------------------------------

    (
        alert_status,
        alert_data,
    ) = post_json(
        f"{ALERT_SERVICE_URL}/alert",
        {
            "breakout_probability":
                probability
        },
    )

    if alert_status != 200:

        raise RuntimeError(
            "Alert Service failed."
        )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Final stock result
    # --------------------------------------------------------

    return {
        "ticker": ticker,
        "date": feature_data["date"],
        "probability": probability,
        "model_prediction":
            model_prediction,
        "threshold":
            alert_data["threshold"],
        "alert":
            alert_data["alert"],
        "decision":
            alert_data["decision"],
        "processing_time":
            processing_time,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    """
    Save batch results for assignment evidence.
    """

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "ticker",
        "date",
        "probability",
        "model_prediction",
        "threshold",
        "alert",
        "decision",
        "processing_time",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# MAIN BATCH EXECUTION
# ============================================================

def main():

    print("=" * 90)

    print(
        "NIFTY 50 - END-TO-END "
        "STOCK BREAKOUT SCREENING"
    )

    print("=" * 90)

    print(
        f"\nStocks requested: "
        f"{len(NIFTY_50)}"
    )

    # --------------------------------------------------------
    # Confirm APIs are available
    # --------------------------------------------------------

    verify_services()

    print("\n" + "=" * 90)

    print("PROCESSING NIFTY 50")

    print("=" * 90)

    start_time = time.perf_counter()

    results = []
    failures = []

    # --------------------------------------------------------
    # Run each stock through all three services
    # --------------------------------------------------------

    for number, ticker in enumerate(
        NIFTY_50,
        start=1,
    ):

        try:

            result = process_stock(
                ticker
            )

            results.append(
                result
            )

            print(
                f"{number:>2}/50 "
                f"{ticker:<12} | "
                f"Prob: "
                f"{result['probability']:.3f} | "
                f"{result['decision']:<20} | "
                f"{result['processing_time']:.3f}s"
            )

        except Exception as error:

            failures.append({
                "ticker": ticker,
                "error": str(error),
            })

            print(
                f"{number:>2}/50 "
                f"{ticker:<12} | "
                f"ERROR | {error}"
            )

    total_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Rank highest probability first
    # --------------------------------------------------------

    results.sort(
        key=lambda result:
            result["probability"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Ranked results
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        "RANKED BREAKOUT CANDIDATES"
    )

    print("=" * 90)

    print(
        f"{'Rank':<6}"
        f"{'Ticker':<14}"
        f"{'Date':<14}"
        f"{'Probability':<15}"
        f"{'Decision'}"
    )

    print("-" * 90)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{result['ticker']:<14}"
            f"{result['date']:<14}"
            f"{result['probability']:<15.3f}"
            f"{result['decision']}"
        )

    # --------------------------------------------------------
    # Alert candidates
    # --------------------------------------------------------

    alerts = [
        result
        for result in results
        if result["alert"] is True
    ]

    print("\n" + "=" * 90)

    print(
        "POTENTIAL BREAKOUT ALERTS"
    )

    print("=" * 90)

    if alerts:

        for result in alerts:

            print(
                f"{result['ticker']:<12} "
                f"Probability: "
                f"{result['probability']:.3f} "
                f"| Threshold: "
                f"{result['threshold']}"
            )

    else:

        print(
            "No stocks exceeded the "
            "configured alert threshold."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        "NIFTY 50 SCREENING SUMMARY"
    )

    print("=" * 90)

    print(
        f"Stocks requested       : "
        f"{len(NIFTY_50)}"
    )

    print(
        f"Successfully processed : "
        f"{len(results)}"
    )

    print(
        f"Failed                 : "
        f"{len(failures)}"
    )

    print(
        f"Potential breakouts    : "
        f"{len(alerts)}"
    )

    print(
        f"Total processing time  : "
        f"{total_time:.3f} seconds"
    )

    if results:

        average_time = (
            total_time
            / len(results)
        )

        print(
            f"Average time per stock : "
            f"{average_time:.3f} seconds"
        )

    # --------------------------------------------------------
    # Display top 5
    # --------------------------------------------------------

    if results:

        print(
            "\nTop 5 stocks by "
            "breakout probability:"
        )

        for rank, result in enumerate(
            results[:5],
            start=1,
        ):

            print(
                f"{rank}. "
                f"{result['ticker']:<12} "
                f"{result['probability']:.3f}"
            )

    # --------------------------------------------------------
    # Failures
    # --------------------------------------------------------

    if failures:

        print(
            "\nStocks that failed:"
        )

        for failure in failures:

            print(
                f"{failure['ticker']} -> "
                f"{failure['error']}"
            )

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    save_results(
        results
    )

    print(
        f"\nBatch results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print("\n" + "=" * 90)

    if len(results) == 50:

        print(
            "SUCCESS: ALL 50 NIFTY STOCKS "
            "PROCESSED END-TO-END"
        )

    else:

        print(
            "WARNING: Not all 50 stocks "
            "were processed successfully."
        )

    print("=" * 90)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nBATCH TEST FAILED:")
        print(error)

        raise