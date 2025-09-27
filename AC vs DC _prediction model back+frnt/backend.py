# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from random import choice, uniform

# -------------------- Fake data generation & model training --------------------
np.random.seed(42)
n_samples = 2000
dates = pd.date_range(start="2024-01-01", periods=n_samples, freq='30min')
hour = dates.hour
day_of_week = dates.dayofweek
is_weekend = (day_of_week >= 5).astype(int)
battery_soc = np.clip(30 + 40*np.sin(2*np.pi*hour/24) + np.random.normal(0,10,n_samples), 5,95)
trip_distance = np.clip(np.random.exponential(25, n_samples),2,200)
location_types = ['Home','Work','Shopping','Highway','Public']
location = np.random.choice(location_types, n_samples, p=[0.4,0.25,0.15,0.1,0.1])
time_urgency = np.random.uniform(1,10,n_samples)
available_time = np.random.uniform(15,480,n_samples)
cost_sensitivity = np.random.uniform(1,10,n_samples)
temperature = 15 + 10*np.sin(2*np.pi*(dates.dayofyear-80)/365) + np.random.normal(0,5,n_samples)

charging_type=[]
for i in range(n_samples):
    score=0
    if battery_soc[i]<20: score+=4
    elif battery_soc[i]<40: score+=2
    if time_urgency[i]>7: score+=3
    elif time_urgency[i]>5: score+=1
    if trip_distance[i]>100: score+=3
    elif trip_distance[i]>50: score+=1
    if location[i]=='Highway': score+=4
    elif location[i]=='Public': score+=2
    elif location[i]=='Home': score-=2
    if available_time[i]<60: score+=3
    elif available_time[i]<120: score+=1
    if cost_sensitivity[i]>7: score-=2
    elif cost_sensitivity[i]>5: score-=1
    score += np.random.normal(0,1)
    charging_type.append('DC' if score>2 else 'AC')

ev_data=pd.DataFrame({
    'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend,
    'battery_soc': battery_soc, 'trip_distance_km': trip_distance,
    'location': location, 'time_urgency': time_urgency,
    'available_time_min': available_time, 'cost_sensitivity': cost_sensitivity,
    'temperature_c': temperature, 'charging_type': charging_type
})

le_location = LabelEncoder()
ev_data['location_encoded'] = le_location.fit_transform(ev_data['location'])
feature_columns = ['hour','day_of_week','is_weekend','battery_soc','trip_distance_km',
                   'location_encoded','time_urgency','available_time_min','cost_sensitivity','temperature_c']
X = ev_data[feature_columns]
y = ev_data['charging_type']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train,y_train)

# -------------------- FastAPI --------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/notification")
def get_notification():
    # Fake current values (could be from GPS, time, or user input)
    current_hour = datetime.now().hour
    current_day = datetime.now().strftime('%A')
    battery = np.clip(np.random.normal(50,20),5,95)
    trip_dist = np.clip(np.random.exponential(30),2,200)
    location_choice = choice(location_types)
    time_urgency_val = np.random.uniform(1,10)
    available_time_val = np.random.uniform(15,480)
    cost_sens = np.random.uniform(1,10)
    temperature_val = np.random.uniform(10,35)
    is_weekend_val = 1 if current_day in ['Saturday','Sunday'] else 0
    day_of_week_val = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(current_day)

    # Encode location
    location_encoded = le_location.transform([location_choice])[0]

    # Prediction
    features = np.array([[current_hour, day_of_week_val, is_weekend_val, battery, trip_dist,
                          location_encoded, time_urgency_val, available_time_val, cost_sens, temperature_val]])
    pred = rf_model.predict(features)[0]
    prob = rf_model.predict_proba(features)[0]
    pred_prob = prob[0] if pred=='AC' else prob[1]

    return {
        "recommendation": pred,
        "confidence": round(pred_prob,2),
        "battery_soc": round(battery,1),
        "trip_distance_km": round(trip_dist,1)
    }
