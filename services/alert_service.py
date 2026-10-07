from fastapi import FastAPI


# ============================================================
# STEP 10 - DECISION / ALERT SERVICE
# ============================================================

ALERT_THRESHOLD = 0.75


# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Stock Breakout Alert Service",
    description="Generates breakout alerts based on prediction probability.",
    version="1.0.0",
)


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "alert-service",
        "threshold": ALERT_THRESHOLD,
    }


# ------------------------------------------------------------
# POST /alert
# ------------------------------------------------------------

@app.post("/alert")
def create_alert(data: dict):

    probability = float(
        data.get("breakout_probability", 0.0)
    )

    if probability >= ALERT_THRESHOLD:

        decision = "Potential Breakout"
        alert = True

    else:

        decision = "No Alert"
        alert = False

    return {
        "breakout_probability": probability,
        "threshold": ALERT_THRESHOLD,
        "alert": alert,
        "decision": decision,
    }