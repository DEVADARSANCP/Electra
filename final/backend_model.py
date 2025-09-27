from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import joblib
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load models (add error handling as needed)
model = tf.keras.models.load_model("ev_energy_model.h5", compile=False)
scaler_seq = joblib.load("scaler_seq.save")
scaler_static = joblib.load("scaler_static.save")
expected_static_features = joblib.load("expected_static_features.save")
one_hot_mapping = joblib.load("one_hot_mapping.save")
seq_features = ['Speed_kmh', 'Acceleration_ms2']
categorical_features = list(one_hot_mapping.keys())

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_energy_adjustment(base_consumption_kwh_per_km: float,
                              traffic_level: str = "Normal",
                              slope_pct: float = 0.0,
                              weather_condition: str = "Sunny"):
    factor = 1.0
    if traffic_level.lower() == "heavy":
        factor *= 1.2
    elif traffic_level.lower() == "moderate":
        factor *= 1.1
    if slope_pct > 0:
        factor *= 1 + slope_pct / 100.0
    elif slope_pct < 0:
        factor *= 1 + slope_pct / 200.0
    if weather_condition.lower() in ["rainy", "snowy", "foggy"]:
        factor *= 1.1
    return base_consumption_kwh_per_km * factor

def compute_multi_leg_trip(trip_legs: list,
                           battery_capacity_kwh: float,
                           current_soc_pct: float,
                           reserve_pct: float = 5.0,
                           avg_speed_kmh: float = None,
                           battery_temp_c: float = None,
                           charger_max_kw: float = 50.0,
                           mae_kwh: float = None):
    results = []
    soc_energy_kwh = (current_soc_pct / 100.0) * battery_capacity_kwh
    reserve_kwh = (reserve_pct / 100.0) * battery_capacity_kwh
    
    for i, leg in enumerate(trip_legs):
        distance = leg.get("distance_km", 0)
        traffic = leg.get("traffic", "Normal")
        slope = leg.get("slope_pct", 0.0)
        weather = leg.get("weather", "Sunny")
        predicted_energy_kwh = leg.get("predicted_energy_kwh", 0)
        
        base_consumption_kwh_per_km = max(predicted_energy_kwh / max(distance, 1e-6), 1e-6)
        adjusted_consumption_kwh_per_km = compute_energy_adjustment(
            base_consumption_kwh_per_km, traffic, slope, weather
        )
        energy_needed_kwh = adjusted_consumption_kwh_per_km * distance
        soc_energy_kwh -= energy_needed_kwh
        soc_energy_kwh_after_reserve = soc_energy_kwh - reserve_kwh
        remaining_range_km = max(0.0, soc_energy_kwh_after_reserve / max(adjusted_consumption_kwh_per_km, 1e-6))
        
        time_to_empty_hours = remaining_range_km / avg_speed_kmh if avg_speed_kmh and avg_speed_kmh > 0 and remaining_range_km > 0 else None
        
        uncertainty_kwh = mae_kwh if mae_kwh is not None else max(0.1, 0.1 * energy_needed_kwh)
        soc_energy_conservative = soc_energy_kwh - uncertainty_kwh
        remaining_range_km_conservative = max(0.0, (soc_energy_conservative - reserve_kwh) / max(adjusted_consumption_kwh_per_km, 1e-6))
        
        needs_charge = soc_energy_kwh_after_reserve < 0
        min_charge_needed_kwh = max(0.0, energy_needed_kwh - (soc_energy_kwh + reserve_kwh))
        
        charging_speed_kw = charger_max_kw
        if battery_temp_c:
            if battery_temp_c > 45: 
                charging_speed_kw *= 0.3
            elif battery_temp_c > 35: 
                charging_speed_kw *= 0.6
        
        charging_time_hours = min_charge_needed_kwh / max(charging_speed_kw, 1e-6) if min_charge_needed_kwh > 0 else None
        
        if needs_charge:
            recommended_action = f"Charge at least {min_charge_needed_kwh:.2f} kWh before starting leg {i+1}"
        elif remaining_range_km < distance:
            recommended_action = "Plan a charging stop along the route"
        else:
            recommended_action = "No charging needed for this leg"
        
        results.append({
            "leg_index": i + 1,
            "distance_km": distance,
            "traffic": traffic,
            "slope_pct": slope,
            "weather": weather,
            "energy_needed_kwh": energy_needed_kwh,
            "remaining_range_km": remaining_range_km,
            "remaining_range_km_conservative": remaining_range_km_conservative,
            "time_to_empty_hours": time_to_empty_hours,
            "min_charge_needed_kwh": min_charge_needed_kwh,
            "charging_speed_kw": charging_speed_kw,
            "charging_time_hours": charging_time_hours,
            "recommended_action": recommended_action,
            "needs_charge": needs_charge
        })
    return results

# Fixed endpoint name to match your request
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Save uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    # Read and process CSV
    df_test = pd.read_csv(file_location)
    df_orig = df_test.copy()

    # Preprocess categorical
    for col in categorical_features:
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(str)
    
    df_test = pd.get_dummies(df_test, columns=[col for col in categorical_features if col in df_test.columns])
    
    for col in categorical_features:
        for cat in one_hot_mapping[col]:
            dummy_col = f"{col}_{cat}"
            if dummy_col not in df_test.columns:
                df_test[dummy_col] = 0

    # Scale and predict
    X_static_df = df_test[expected_static_features]
    X_seq = scaler_seq.transform(df_test[seq_features])
    X_static = scaler_static.transform(X_static_df)
    X_seq = X_seq.reshape((X_seq.shape[0], 1, X_seq.shape[1]))
    
    y_pred = model.predict([X_seq, X_static])
    df_orig['Predicted_Energy_KWh'] = y_pred.flatten()

    # Create trip legs
    trip_legs = []
    for _, row in df_orig.iterrows():
        trip_legs.append({
            "distance_km": row.get("Distance_Travelled_km", 0),
            "traffic": row.get("Traffic_Condition", "Normal"),
            "slope_pct": row.get("Slope_%", 0),
            "weather": row.get("Weather_Condition", "Sunny"),
            "predicted_energy_kwh": row.get("Predicted_Energy_KWh", 0)
        })

    # Compute trip plan
    battery_capacity = 40
    current_soc = df_orig["Battery_State_%"].iloc[0]
    avg_speed = df_orig["Speed_kmh"].iloc[0]
    battery_temp = df_orig["Battery_Temperature_C"].iloc[0]

    plan = compute_multi_leg_trip(
        trip_legs=trip_legs,
        battery_capacity_kwh=battery_capacity,
        current_soc_pct=current_soc,
        avg_speed_kmh=avg_speed,
        battery_temp_c=battery_temp
    )

    # Save and return result.csv
    df_plan = pd.DataFrame(plan)
    result_file = os.path.join(UPLOAD_DIR, "result.csv")
    df_plan.to_csv(result_file, index=False)

    return FileResponse(result_file, filename="result.csv", media_type="text/csv")

# Also keep the original endpoint for backward compatibility
@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...)):
    return await upload_csv(file)

@app.get("/")
def root():
    return {"message": "EV Energy Prediction - CSV Processing Server Running"}