"""
train_model.py
----------------
Trains the delay-prediction model and saves it (with the preprocessing
pipeline) to model.pkl so the FastAPI backend can load it directly.

Also saves feature_importance.json which the backend uses to power the
"Delay Cause Analysis" explainability feature — instead of a black-box
prediction, the API can say *why* the model expects a delay.

Run:
    python generate_data.py
    python train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("train_delays.csv")

CATEGORICAL = ["train_no", "source", "destination", "weather", "congestion"]
NUMERIC = [
    "day_of_week",
    "is_weekend",
    "distance_km",
    "historical_avg_delay",
    "technical_issue",
    "signal_issue",
]

X = df[CATEGORICAL + NUMERIC]
y = df["delay_minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ],
    remainder="passthrough",
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)),
    ]
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f} minutes")
print(f"R2 score: {r2:.3f}")

joblib.dump(model, "model.pkl")

# ---- Explainability: aggregate feature importance back to human-readable groups
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL)
all_feature_names = list(cat_feature_names) + NUMERIC
importances = model.named_steps["regressor"].feature_importances_

importance_map = {}
for name, score in zip(all_feature_names, importances):
    group = name.split("_")[0] if "_" not in name[:8] else name.split("_")[0]
    # group raw categorical columns back to their original column name
    for col in CATEGORICAL:
        if name.startswith(col):
            group = col
            break
    else:
        group = name
    importance_map[group] = importance_map.get(group, 0) + float(score)

total = sum(importance_map.values())
normalized = {k: round((v / total) * 100, 1) for k, v in importance_map.items()}
normalized = dict(sorted(normalized.items(), key=lambda kv: -kv[1]))

with open("feature_importance.json", "w") as f:
    json.dump(normalized, f, indent=2)

print("\nFeature contribution to delay (%):")
for k, v in normalized.items():
    print(f"  {k:25s} {v}%")

print("\nSaved model.pkl + feature_importance.json")
