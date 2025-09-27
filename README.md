# EV Smart Trip Planner

> A hackathon project that reimagines EV trip planning — reducing range anxiety, optimizing charging stops, and turning downtime into meaningful time.

---

##  Problem

With EV adoption rising, drivers face new challenges:

- **Range Anxiety** → Will I make it to my destination?
- **Battery Degradation** → Every EV’s usable capacity differs from spec sheets.
- **Wasted Charging Time** → Drivers scroll or sit idle instead of making the most of stopovers.

---

## Our Solution

**EV Smart Trip Planner** is like Google Maps but designed for EV owners.

### Core Features

1. **EV-Aware Route Planning**
   - Suggests routes considering charging stops, car model, battery %, and driving style.

2. **Battery Health Integration**
   - Estimates real usable capacity via State of Health (SoH).
   - Factors in car age, mileage, and efficiency.

3. **Charging Stop Optimization**
   - Calculates charging times based on charger type & battery size.
   - Sends notifications when ready to continue.

4. **Wellness & Family Experience**
   - Recommends activities during charging:
     - Stretch/exercise routines 🏃
     - Cafes & local attractions ☕
     - Family games 👨‍👩‍👧‍👦

---

## 🛠️Tech Stack

**Frontend**
- React (Web) / React Native (Mobile)
- TailwindCSS for styling

**Backend / APIs**
- Node.js (Express) or Firebase
- Google Maps API / Mapbox → routing & POIs
- Open Charge Map API → charging station data
- Mock EV Battery Data / Tesla API (if accessible)

**Data & Features**
- EV Database API → specs (battery, range, efficiency)
- Custom SoH estimation module
- Firebase push notifications

---

## Challenges

- **Battery Health Access** → true SoH requires OBD-II or OEM APIs → workaround with estimates.
- **Driving Behavior Variability** → speed, terrain, AC use → modeled with a “driving style” slider.
- **Charger Types** → varied power outputs → simplified with average charging times.
- **Data Availability** → some EV APIs are closed → use mock/demo data for hackathon.

---

## MVP Scope (Hackathon Deliverable)

- User selects **Start & Destination**.
- App suggests **route + charging stops**.
- Displays **estimated range** (EV model + health factor).
- Shows **time to charge (to 80% / full)**.
- Sends **charging complete notifications**.
- Recommends **activities during stopovers**.

---

##  Pitch Line

> *“Our app doesn’t just solve range anxiety — it makes EV road trips smarter, healthier, and more meaningful. By combining EV-specific trip planning with lifestyle recommendations, we transform downtime into quality time.”*

---

##  Demo Preview (to add)

- Route planning screenshot
- Charging stop suggestion card
- Wellness recommendation popup

---

## Team

- Electra
- Built for HashItUp, 2025

---
