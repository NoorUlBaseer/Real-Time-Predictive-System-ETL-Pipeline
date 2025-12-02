from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import joblib
import pandas as pd

app = FastAPI(title="Stock Sentiment Predictor")  # FastAPI instance

# PROMETHEUS SETUP
instrumentator = Instrumentator().instrument(app).expose(app)  # Auto-instrumentation for Prometheus

# Custom Metrics
TOTAL_PREDICTIONS = Counter("total_predictions", "Total number of inference requests")  # Counter for total rows processed
DRIFTED_PREDICTIONS = Counter("drifted_predictions", "Number of requests flagged as OOD")  # Counter for 'drifted' rows
DRIFT_RATIO = Gauge("data_drift_ratio", "Ratio of OOD requests to total requests")  # Gauge for the ratio (Drift / Total)

MODEL_PATH = "model_dir/stock_sentiment_model.pkl"  # Path to the trained model
try:
    model = joblib.load(MODEL_PATH)  # Load the trained model
    print("✅ Model loaded successfully.")
except Exception as e:  # Catch all exceptions during model loading
    print(f"❌ Failed to load model: {e}")
    model = None


class NewsInput(BaseModel):  # Pydantic model for input validation
    hour_of_day: int
    day_of_week: int
    title_sentiment: float
    desc_sentiment: float
    content_sentiment: float


def check_drift(data):  # Simple drift detection logic
    is_outlier = False
    # If sentiment is exactly -1 or 1, we call it "suspicious/drifted" for this demo
    if abs(data['title_sentiment'].iloc[0]) > 0.95:
        is_outlier = True

    return is_outlier


@app.get("/")
def health_check():  # Health check endpoint
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(news: NewsInput):  # Prediction endpoint
    if not model:  # Model not loaded
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([news.dict()])  # Convert input to DataFrame

    TOTAL_PREDICTIONS.inc()  # Increment total predictions counter

    if check_drift(data):  # Check for drift
        DRIFTED_PREDICTIONS.inc()  # Increment drifted predictions counter

    total = TOTAL_PREDICTIONS._value.get()  # Get current total predictions
    drifted = DRIFTED_PREDICTIONS._value.get()  # Get current drifted predictions

    if total > 0:  # Avoid division by zero
        DRIFT_RATIO.set(drifted / total)  # Update drift ratio gauge

    try:
        prediction = model.predict(data)[0]  # Make prediction
        return {
            "predicted_market_impact": float(prediction),  # Return prediction as float
            "interpretation": "Market UP" if prediction > 0 else "Market DOWN"  # Simple interpretation
        }
    except Exception as e:  # Handle prediction errors
        raise HTTPException(status_code=500, detail=str(e))
