from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime

app = FastAPI(title="Food Delivery Time Prediction API")

# Load saved model, scaler and training columns
model = joblib.load("xgb_best.pkl")
scaler = joblib.load("standard_scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# Input data
class DeliveryInput(BaseModel):
    Delivery_person_Age: float
    Delivery_person_Ratings: float

    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float

    order_time: str

    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str


# Calculate distance using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):

    r = 6371

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return r * c


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict(data: DeliveryInput):

    # Calculate distance
    distance_km = calculate_distance(
        data.Restaurant_latitude,
        data.Restaurant_longitude,
        data.Delivery_location_latitude,
        data.Delivery_location_longitude
    )

    # Extract order hour
    try:
        order_hour = datetime.strptime(
            data.order_time, "%H:%M:%S"
        ).hour
    except ValueError:
        return {"error": "order_time format must be HH:MM:SS"}

    # Traffic encoding
    traffic_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Jam": 3
    }

    if data.Road_traffic_density not in traffic_mapping:
        return {"error": "Invalid road traffic density"}

    # Prepare numerical features
    input_data = {
        "Delivery_person_Age": data.Delivery_person_Age,
        "Delivery_person_Ratings": data.Delivery_person_Ratings,
        "distance_km": distance_km,
        "order_hour": order_hour,
        "Road_traffic_density": traffic_mapping[data.Road_traffic_density],
        "Vehicle_condition": data.Vehicle_condition,
        "multiple_deliveries": data.multiple_deliveries
    }

    # Categorical features
    categories = {
        "Weatherconditions": data.Weatherconditions,
        "Type_of_order": data.Type_of_order,
        "Type_of_vehicle": data.Type_of_vehicle,
        "Festival": data.Festival,
        "City": data.City
    }

    # Create row using exact training feature columns
    row = {col: 0 for col in model_columns}

    # Add numerical features
    row.update(input_data)

    # Set one-hot encoded features
    for feature, value in categories.items():
        column_name = f"{feature}_{value}"

        if column_name in row:
            row[column_name] = 1

    # Arrange columns in training order
    input_df = pd.DataFrame([row], columns=model_columns)

    # Apply saved scaler
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    return {
        "predicted_delivery_time": round(float(prediction), 2),
        "unit": "minutes"
    }