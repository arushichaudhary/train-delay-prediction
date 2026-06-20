# Velora — Intelligent Rail Delay & Analytics Platform

Velora predicts train delays, explains *why* a delay is likely, suggests
alternatives, and visualizes network-wide congestion and historical
performance — a full product, not just a notebook with a `.predict()` call.


## 1. What's inside

```
velora/
├── frontend/
│   └── index.html          # Single-file dashboard (Tailwind CDN, no build step)
├── backend/
│   ├── app.py               # FastAPI service — predict, analytics, chatbot, etc.
│   ├── requirements.txt
│   └── velora.db            # SQLite (auto-created on first run)
├── ml/
│   ├── generate_data.py     # Synthetic-but-realistic training data generator
│   ├── train_model.py       # Trains RandomForestRegressor + saves model.pkl
│   ├── train_delays.csv     # Generated dataset (12,000 rows)
│   ├── model.pkl            # Trained model (already included, ready to use)
│   └── feature_importance.json
└── docs/
    └── ARCHITECTURE.md
```

## 2. Quick start

### Run the backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # then fill in VELORA_SECRET_KEY and (optionally) GEMINI_API_KEY
uvicorn app:app --reload --port 8000
```
The API will be live at `http://localhost:8000`. Visit `/docs` for the
interactive Swagger UI (FastAPI generates this automatically).

**Generate a secret key:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Get a free Gemini key** (optional — chatbot works without it, just with a
rule-based fallback reply): https://aistudio.google.com/app/apikey

### Open the frontend
Just open `frontend/index.html` in a browser. It auto-detects the backend:
- If `http://localhost:8000` is reachable → uses real predictions from your
  trained model.
- If not → falls back to realistic mock data, so the **UI is fully demoable
  even with zero setup** (useful for a quick interview screen-share).

### Re-train the model (optional)
```bash
cd ml
python generate_data.py   # regenerate the synthetic dataset
python train_model.py     # retrain + overwrite model.pkl
```
Current model: **RandomForestRegressor**, MAE ≈ 7.1 minutes, R² ≈ 0.85 on
held-out data.

## 3. Feature → endpoint map

| Feature                         | Endpoint                          |
|----------------------------------|------------------------------------|
| Delay Prediction Engine          | `POST /api/predict`               |
| Live Train Dashboard             | `GET /api/trains`                 |
| Delay Cause Analysis (XAI)       | included in `/api/predict` response (`causes`) |
| Alternative Train Recommendation | included in `/api/predict` response (`alternatives`, triggers when delay > 60 min) |
| Route Congestion Heatmap         | `GET /api/congestion`             |
| Weather Impact Analysis          | feature-importance values from the trained model, exposed via `/api/predict` causes |
| AI Travel Assistant (chatbot)    | `POST /api/chatbot`               |
| Personalized Notifications       | `POST/GET /api/saved-trains`      |
| Delay Analytics Dashboard        | `GET /api/analytics/overview`     |
| Railway Performance Ranking      | included in `/api/analytics/overview` (`most_punctual_routes`, `most_delayed_routes`) |
| Crowd Reports                    | `POST/GET /api/crowd-reports`     |
| Sentiment Analysis               | `GET /api/sentiment`              |
| Delay Simulator                  | `POST /api/simulate`              |
| Admin Dashboard stats            | `GET /api/admin/stats`            |

## 4. How the ML model works (for interviews)

1. **`generate_data.py`** builds a synthetic dataset where delay is a
   function of weather, congestion, distance, historical average delay,
   and incident flags (technical/signal issues) — with realistic noise.
   In a production version this would be replaced by scraped/NTES/IRCTC
   data + a weather API.
2. **`train_model.py`** one-hot encodes categorical columns (train number,
   stations, weather, congestion) inside a `ColumnTransformer`, then fits a
   `RandomForestRegressor` inside a `sklearn.Pipeline` so preprocessing and
   model travel together as one artifact (`model.pkl`).
3. **Explainability**: feature importances from the forest are aggregated
   back to human-readable groups (weather, congestion, technical issue,
   etc.) and saved to `feature_importance.json`. The backend turns these
   into the "Why this delay?" cause list — this is what makes the project
   stand out over a plain regression demo.
4. **`/api/simulate`** re-runs the same model with user-edited inputs
   (weather, congestion, issue flags) for the "Delay Simulator" feature.

## 5. What to say in the interview

- *"I didn't just train a regression model — I shipped it behind a REST
  API and built explainability into the response, because raw delay
  numbers without a reason aren't actionable."*
- *"The frontend works standalone with mock data so it's demoable anywhere,
  and switches to live predictions the moment the API is reachable — that
  was a deliberate resilience decision."*
- *"Performance ranking and congestion heatmaps turn this from a single
  prediction tool into a small analytics product, which is what the
  'platform' part of the name is about."*

## 6. What was fixed in this revision

- **Deterministic predictions**: weather/congestion/incident flags are now
  sampled from a `random.Random` seeded by a hash of `(train_no, source,
  destination, date)`, not the global RNG. Same input → same output, every
  time — required for a credible demo and for caching.
- **JWT auth**: real signup/login (`/api/auth/signup`, `/api/auth/login`),
  bcrypt password hashing via `passlib`, and `/api/saved-trains` now
  requires a valid bearer token (`Depends(get_current_user)`) instead of
  trusting a raw email string in the request body.
- **Gemini-backed chatbot**: `/api/chatbot` calls the Gemini API with the
  model's real predicted delay as grounding context (so it explains an
  actual number instead of inventing one). Falls back to a rule-based
  reply if `GEMINI_API_KEY` isn't set, so the feature never hard-fails a
  demo.
- **Deployment configs**: `render.yaml` + `Procfile` for the backend,
  step-by-step guide in `docs/DEPLOYMENT.md` for backend (Render/Railway)
  + frontend (Vercel/Netlify).

## 7. Remaining honest limitations

- Live running-status, weather, and congestion are still
  simulated/synthetic — wiring up the NTES/IRCTC unofficial APIs and a real
  weather API (e.g. OpenWeatherMap) is the natural next step.
- Sentiment analysis still returns randomized labels as a placeholder for a
  real NLP pipeline (e.g. VADER or a fine-tuned model over scraped tweets).
- CORS is wide open (`allow_origins=["*"]`) for easy local demoing — lock
  this to your real frontend domain before calling it production (see
  `docs/DEPLOYMENT.md`).
- No password-reset flow, no email verification, no rate limiting on auth
  endpoints — fine for a resume project, not fine for real users.

## 8. Next steps to push it further

- Swap synthetic data for a real historical delay dataset (Kaggle has a
  few Indian Railways delay datasets) and retrain.
- Add a real weather API call in `run_prediction()`.
- Add JWT auth + per-user notification preferences (email/SMS via a
  provider like Twilio/SendGrid).
- Deploy backend on Render/Railway and frontend on Vercel/Netlify for a
  live demo link on your resume.
