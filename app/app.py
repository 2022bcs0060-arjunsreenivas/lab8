from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="California Housing Prediction API")

# Load trained model
model = joblib.load("model.pkl")


class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: int


@app.get("/")
def home():
    return {"message": "California Housing Price Prediction API"}

@app.get("/health")
def health():
    return {"message":"eda healthy kutta"}

@app.post("/predict")
def predict(features: HouseFeatures):
    
    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)

    return {"predicted_price": float(prediction[0])}