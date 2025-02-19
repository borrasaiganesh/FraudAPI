from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: dict):
    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return result
    return {"fraud": bool(prediction[0])}

# Run this API with: uvicorn filename:app --reload
