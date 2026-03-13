import os
from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np


class EVInput(BaseModel):
    top_speed_kmh: float
    battery_capacity_kWh: float
    torque_nm: float
    acceleration_0_100_s: float
    fast_charging_power_kw_dc: float


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "RF_regressor.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
app = FastAPI(title="EV range prediction")

@app.get("/")
def status():
    return {"status":"Running!"}

@app.post("/predict")
def predict(data: EVInput):
    try:
        features = np.array([[
            data.top_speed_kmh,
            data.battery_capacity_kWh,
            data.torque_nm,
            data.acceleration_0_100_s,
            data.fast_charging_power_kw_dc
        ]])
        scaled_input = scaler.transform(features)
        prediction = model.predict(scaled_input)[0]
        return {
                "status": "success",
                "predicted_range_km": round(float(prediction),2)
                }
        
    except Exception as e:
        return {
            "status":"error",
            "message":str(e) 
        }