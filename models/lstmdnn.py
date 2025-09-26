import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

#load dataset
df = pd.read_csv("EV_Energy_Consumption_Dataset.csv")

#features 
seq_features = ['Speed_kmh', 'Acceleration_ms2']

categorical_features = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
df = pd.get_dummies(df, columns=categorical_features)

static_features = [
    'Battery_State_%', 'Battery_Voltage_V', 'Battery_Temperature_C', 'Slope_%', 
    'Temperature_C', 'Humidity_%', 'Wind_Speed_ms', 'Tire_Pressure_psi', 
    'Vehicle_Weight_kg', 'Distance_Travelled_km'
] + [col for col in df.columns if col.startswith(tuple(categorical_features))]

target = 'Energy_Consumption_kWh'

# Fill missing values

df[seq_features + static_features] = df[seq_features + static_features].fillna(0)
df[target] = df[target].fillna(0)

# Scale numerical features

scaler_seq = StandardScaler()
X_seq = scaler_seq.fit_transform(df[seq_features])

scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(df[static_features])

y = df[target].values
#train test split
X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_seq, X_static, y, test_size=0.2, random_state=42
)
# Reshape sequence for LSTM (samples, timesteps, features)
# Here we treat each row as a single timestep

X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], 1, X_seq_train.shape[1]))
X_seq_test = X_seq_test.reshape((X_seq_test.shape[0], 1, X_seq_test.shape[1]))

# Build LSTM branch

seq_input = Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]), name="seq_input")
x = LSTM(32, activation='tanh')(seq_input)
x = Dense(16, activation='relu')(x)

# Build DNN branch
static_input = Input(shape=(X_static_train.shape[1],), name="static_input")
y_static = Dense(64, activation='relu')(static_input)
y_static = Dense(32, activation='relu')(y_static)

# Combine branches
combined = Concatenate()([x, y_static])
z = Dense(32, activation='relu')(combined)
z = Dense(16, activation='relu')(z)
output = Dense(1, activation='linear')(z)

model = Model(inputs=[seq_input, static_input], outputs=output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train model

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [X_seq_train, X_static_train], y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate model

loss, mae = model.evaluate([X_seq_test, X_static_test], y_test, verbose=2)
print(f"Test MAE: {mae:.4f}")

# Save model and scalers

model.save("ev_energy_model.h5")
import joblib
joblib.dump(scaler_seq, "scaler_seq.save")
joblib.dump(scaler_static, "scaler_static.save")

print("Model and scalers saved")
