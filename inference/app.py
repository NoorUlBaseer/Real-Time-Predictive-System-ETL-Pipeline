from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Stock Sentiment Predictor")  # API instance

MODEL_PATH = "model_dir/stock_sentiment_model.pkl"  # Path to the downloaded model

try:
    model = joblib.load(MODEL_PATH)  # Load the trained model
    print("✅ Model loaded successfully.")
except Exception as e:  # Handle exceptions during model loading
    print(f"❌ Failed to load model: {e}")
    model = None


class NewsInput(BaseModel):  # Define input schema for news data
    hour_of_day: int
    day_of_week: int
    title_sentiment: float
    desc_sentiment: float
    content_sentiment: float


@app.get("/")
def health_check():  # Health check endpoint
    if model:  # Model loaded successfully
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "error": "Model not loaded"}  # Model not loaded


@app.post("/predict")
def predict(news: NewsInput):  # Prediction endpoint
    if not model:  # if model is not loaded, raise an error
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([news.dict()])  # Convert input data to DataFrame

    try:
        prediction = model.predict(data)[0]  # Make prediction using the model
        return {
            "predicted_market_impact": float(prediction),  # Convert prediction to float
            "interpretation": "Market UP" if prediction > 0 else "Market DOWN"  # Interpret prediction
        }
    except Exception as e:  # Handle prediction errors
        raise HTTPException(status_code=500, detail=str(e))
