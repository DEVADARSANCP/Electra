# backend_model.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import tensorflow as tf
import os

# -------------------------
# Load model and preprocessing objects once
# -------------------------
model = tf.keras.models.load_model("ev_energy_model.h5", compile=False)
scaler_seq = joblib.load("scaler_seq.save")
scaler_static = joblib.load("scaler_static.save")
expected_static_features = joblib.load("expected_static_features.save")
one_hot_mapping = joblib.load("one_hot_mapping.save")

seq_features = ['Speed_kmh', 'Acceleration_ms2']
categorical_features = list(one_hot_mapping.keys())

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "EV Route Energy Prediction Backend Running"}

# -------------------------
# Endpoint 1: Receive CSV and predict energy
# -------------------------
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load CSV
        df_test = pd.read_csv(file_location)
        df_orig = df_test.copy()

        # Ensure categorical columns are strings
        for col in categorical_features:
            df_test[col] = df_test[col].astype(str)

        # One-hot encode categorical columns
        df_test = pd.get_dummies(df_test, columns=categorical_features)

        # Add missing dummy columns from training
        for col in categorical_features:
            for cat in one_hot_mapping[col]:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in df_test.columns:
                    df_test[dummy_col] = 0

        # Reorder static features to match training
        X_static_df = df_test[expected_static_features]

        # Scale features
        X_seq = scaler_seq.transform(df_test[seq_features])
        X_static = scaler_static.transform(X_static_df)

        # Reshape sequence for LSTM
        X_seq = X_seq.reshape((X_seq.shape[0], 1, X_seq.shape[1]))

        # Predict
        y_pred = model.predict([X_seq, X_static])

        # Add prediction to original dataframe
        df_orig['Predicted_Energy_kWh'] = y_pred

        # Save predictions
        pred_file = os.path.join(UPLOAD_DIR, f"prediction_{file.filename}")
        df_orig.to_csv(pred_file, index=False)

        return JSONResponse({
            "status": "success",
            "saved_path": pred_file,
            "predictions": y_pred.tolist()
        })

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)})

# -------------------------
# Endpoint 2: Process predictions CSV
# -------------------------
@app.post("/process-predictions/")
async def process_predictions(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load CSV
        df = pd.read_csv(file_location)

        # -------------------------
        # Your second Python code logic here
        # Example: Add new column, do some calculations, etc.
        # -------------------------
        # Example: add 10 kWh to predicted energy
        df['Energy_kWh_plus_10'] = df['Predicted_Energy_kWh'] + 10

        # Save processed CSV
        processed_file = os.path.join(UPLOAD_DIR, f"processed_{file.filename}")
        df.to_csv(processed_file, index=False)

        return JSONResponse({
            "status": "success",
            "processed_file": processed_file,
            "example_result": df.head(3).to_dict(orient="records")  # preview
        })

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)})
