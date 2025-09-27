import math

def compute_energy_adjustment(base_consumption_kwh_per_km: float,
                              traffic_level: str = "Normal",
                              slope_pct: float = 0.0,
                              weather_condition: str = "Sunny"):
    factor = 1.0
    # Traffic adjustment
    if traffic_level.lower() == "heavy":
        factor *= 1.2
    elif traffic_level.lower() == "moderate":
        factor *= 1.1

    # Slope adjustment (increase consumption for uphill)
    if slope_pct > 0:
        factor *= 1 + slope_pct / 100.0
    elif slope_pct < 0:
        factor *= 1 + slope_pct / 200.0  # downhill reduces slightly

    # Weather adjustment
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

        # Base consumption per km
        base_consumption_kwh_per_km = max(predicted_energy_kwh / max(distance, 1e-6), 1e-6)

        # Adjust for traffic, slope, and weather
        adjusted_consumption_kwh_per_km = compute_energy_adjustment(
            base_consumption_kwh_per_km, traffic, slope, weather
        )

        # Energy needed for this leg
        energy_needed_kwh = adjusted_consumption_kwh_per_km * distance

        # Energy remaining after leg
        soc_energy_kwh -= energy_needed_kwh
        soc_energy_kwh_after_reserve = soc_energy_kwh - reserve_kwh

        # Remaining range for leg (cannot be negative)
        remaining_range_km = max(0.0, soc_energy_kwh_after_reserve / adjusted_consumption_kwh_per_km)

        # Time to empty
        time_to_empty_hours = None
        if avg_speed_kmh and avg_speed_kmh > 0 and remaining_range_km > 0:
            time_to_empty_hours = remaining_range_km / avg_speed_kmh

        # Conservative estimate
        uncertainty_kwh = mae_kwh if mae_kwh is not None else max(0.1, 0.1 * energy_needed_kwh)
        soc_energy_conservative = soc_energy_kwh - uncertainty_kwh
        remaining_range_km_conservative = max(0.0, (soc_energy_conservative - reserve_kwh) / adjusted_consumption_kwh_per_km)

        # Check if charging is needed
        needs_charge = soc_energy_kwh_after_reserve < 0
        min_charge_needed_kwh = max(0.0, energy_needed_kwh - (soc_energy_kwh + reserve_kwh))

        # Adjust charging speed for battery temperature
        charging_speed_kw = charger_max_kw
        if battery_temp_c is not None:
            if battery_temp_c > 45:
                charging_speed_kw *= 0.3
            elif battery_temp_c > 35:
                charging_speed_kw *= 0.6

        charging_time_hours = None
        if min_charge_needed_kwh > 0 and charging_speed_kw > 0:
            charging_time_hours = min_charge_needed_kwh / charging_speed_kw

        # Recommendation
        if needs_charge:
            recommended_action = f"Charge at least {min_charge_needed_kwh:.2f} kWh before starting leg {i+1}"
        elif remaining_range_km < distance:
            recommended_action = "Plan a charging stop along the route"
        else:
            recommended_action = "No charging needed for this leg"

        # Save leg results
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


# Example usage
if __name__ == "__main__":
    trip_legs = [
        {"distance_km": 50, "traffic": "Heavy", "slope_pct": 2, "weather": "Sunny", "predicted_energy_kwh": 10},
        {"distance_km": 30, "traffic": "Moderate", "slope_pct": -1, "weather": "Rainy", "predicted_energy_kwh": 6}
    ]

    plan = compute_multi_leg_trip(
        trip_legs=trip_legs,
        battery_capacity_kwh=40,
        current_soc_pct=45,
        avg_speed_kmh=50,
        battery_temp_c=40
    )

    for leg in plan:
        print(f"Leg {leg['leg_index']}:")
        for k, v in leg.items():
            if k != "leg_index":
                print(f"  {k}: {v}")
        print("\n")
