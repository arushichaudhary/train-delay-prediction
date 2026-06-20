"""
generate_data.py
-----------------
Creates a synthetic but realistic train-delay dataset so the project can be
trained and demoed end-to-end without needing a live data-pipeline / scraping
setup (which is usually the hardest part to get working before a placement
interview).

Run:
    python generate_data.py

Produces: ml/train_delays.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 12000

TRAINS = [
    ("12951", "Mumbai Rajdhani Express"),
    ("12301", "Howrah Rajdhani Express"),
    ("12009", "Shatabdi Express"),
    ("22439", "Vande Bharat Express"),
    ("12259", "Sealdah Duronto Express"),
    ("12626", "Kerala Express"),
    ("12723", "Telangana Express"),
    ("12841", "Coromandel Express"),
    ("12471", "Swaraj Express"),
    ("12903", "Golden Temple Mail"),
]

STATIONS = [
    "New Delhi", "Delhi Junction", "Kanpur Central", "Mughal Sarai",
    "Howrah Jn", "Mumbai Central", "Vadodara Jn", "Surat",
    "Chennai Central", "Vijayawada Jn", "Nagpur Jn", "Bhopal Jn",
    "Lucknow NR", "Patna Jn", "Kota Jn", "Ajmer Jn",
]

WEATHER = ["Clear", "Light Rain", "Heavy Rain", "Fog", "Storm"]
WEATHER_DELAY_FACTOR = {"Clear": 0, "Light Rain": 8, "Heavy Rain": 25, "Fog": 35, "Storm": 30}

rows = []
for _ in range(N):
    train_no, train_name = TRAINS[np.random.randint(len(TRAINS))]
    src, dst = np.random.choice(STATIONS, 2, replace=False)
    weather = np.random.choice(WEATHER, p=[0.45, 0.2, 0.15, 0.12, 0.08])
    congestion = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.32, 0.18])
    congestion_factor = {"Low": 0, "Medium": 12, "High": 28}[congestion]
    day_of_week = np.random.randint(0, 7)
    is_weekend = 1 if day_of_week >= 5 else 0
    distance_km = np.random.randint(150, 2200)
    hist_avg_delay = np.clip(np.random.normal(18, 12), 0, 90)
    technical_issue = np.random.choice([0, 1], p=[0.92, 0.08])
    signal_issue = np.random.choice([0, 1], p=[0.93, 0.07])

    base = 4 + 0.01 * distance_km
    weather_component = WEATHER_DELAY_FACTOR[weather] * np.random.uniform(0.6, 1.3)
    congestion_component = congestion_factor * np.random.uniform(0.7, 1.3)
    hist_component = hist_avg_delay * 0.35
    incident_component = technical_issue * np.random.uniform(20, 60) + signal_issue * np.random.uniform(15, 45)
    weekend_component = is_weekend * np.random.uniform(-3, 6)
    noise = np.random.normal(0, 6)

    delay_minutes = max(
        0,
        base + weather_component + congestion_component + hist_component
        + incident_component + weekend_component + noise,
    )

    rows.append(
        {
            "train_no": train_no,
            "train_name": train_name,
            "source": src,
            "destination": dst,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "distance_km": distance_km,
            "weather": weather,
            "congestion": congestion,
            "historical_avg_delay": round(hist_avg_delay, 1),
            "technical_issue": technical_issue,
            "signal_issue": signal_issue,
            "delay_minutes": round(delay_minutes, 1),
        }
    )

df = pd.DataFrame(rows)
df.to_csv("train_delays.csv", index=False)
print(f"Generated {len(df)} rows -> ml/train_delays.csv")
print(df.head())
