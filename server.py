from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('app/model.joblib')
le = joblib.load('app/label_encoder.joblib')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
@app.get('/')
def home():
    return {"message": "Crop Prediction API - Send GET request to /predict"}

@app.get('/predict')
def predict(n: float, p: float, k: float, temp: float, 
            humidity: float, ph: float, rainfall: float):
    """
    Predict top-3 crops with probabilities
    """
    # Create feature array in correct order
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
    # Get probability predictions for all classes
    probabilities = model.predict_proba(features)[0]  # Get first (and only) sample
    
    # Get top-3 indices
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    
    # Get top-3 probabilities and crop names
    top_3_probs = probabilities[top_3_indices]
    top_3_crops = le.inverse_transform(top_3_indices)
    
    # Format response
    predictions = [
        {
            "rank": i + 1,
            "crop": crop,
            "probability": float(prob),
            "confidence_percentage": float(prob * 100)
        }
        for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs))
    ]
    
    return {
        "top_predictions": predictions,
        "recommended_crop": top_3_crops[0],  # Top recommendation
        "input_parameters": {
            "N": n, "P": p, "K": k,
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
    }