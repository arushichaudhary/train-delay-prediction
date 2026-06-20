# Velora — Architecture

```
┌──────────────────────────┐        ┌──────────────────────────────┐
│       frontend/           │  HTTP  │           backend/            │
│   index.html (Tailwind)   │ <----> │   FastAPI app.py               │
│  - Predict form            │        │   - /api/predict               │
│  - Live dashboard           │        │   - /api/simulate              │
│  - Congestion heatmap        │        │   - /api/congestion            │
│  - Analytics charts (SVG)     │        │   - /api/analytics/overview    │
│  - AI chat widget               │        │   - /api/chatbot               │
│  - Crowd report form              │        │   - /api/saved-trains          │
│                                      │        │   - /api/crowd-reports         │
│  Falls back to mock data if           │        │   - /api/sentiment             │
│  backend is unreachable                 │        │   - /api/admin/stats           │
└──────────────────────────┘        └───────────────┬──────────────┘
                                                       │
                                          loads at startup
                                                       │
                                     ┌─────────────────▼─────────────────┐
                                     │              ml/                   │
                                     │  generate_data.py → train_delays.csv │
                                     │  train_model.py  → model.pkl         │
                                     │                  → feature_importance.json │
                                     └─────────────────────────────────────┘
                                                       │
                                         RandomForestRegressor
                                      (sklearn Pipeline: OneHotEncoder + RF)
```

## Request flow: a single prediction

1. User submits the predict form (train number, source, destination, date).
2. Frontend `POST`s JSON to `/api/predict`.
3. Backend builds a feature row (weather/congestion are sampled the way
   they would be looked up from a live weather/congestion service), then
   calls `model.predict()`.
4. Backend converts the model's raw feature importances into a
   human-readable "causes" list (Delay Cause Analysis).
5. If predicted delay > 60 minutes, backend attaches 3 alternative train
   suggestions.
6. Frontend renders the result as a "ticket" card with a split-flap delay
   counter, cause list, and alternatives.

## Why a Pipeline, not a raw model

`ColumnTransformer` (OneHotEncoder) + `RandomForestRegressor` are bundled
into one `sklearn.Pipeline` and pickled together. This means the backend
never has to remember "which columns were one-hot encoded in what order" —
it just calls `.predict()` on a raw DataFrame with the original column
names. This is the same pattern used in production ML services and is
worth mentioning in interviews.

## Why SQLite, not Postgres

For a placement project, SQLite removes any setup friction (no DB server
to run) while still demonstrating real persistence (saved trains, crowd
reports) and a clean path to swap in Postgres later — just change the
connection string, the schema stays the same.
