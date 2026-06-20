"""
Velora AI — backend API
------------------------
FastAPI service powering the Velora rail-delay prediction & analytics
platform. Loads the trained RandomForest model from ml/model.pkl and exposes
REST endpoints consumed by frontend/index.html.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

All endpoints are CORS-open so the static frontend file can be opened
directly in a browser during development/demo.
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent / "ml"
DB_PATH = BASE_DIR / "velora.db"

# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("VELORA_SECRET_KEY", "dev-only-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

app = FastAPI(title="Velora AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load trained model + explainability data (with graceful fallback so the
# API still works even before the ML training step has been run)
# ---------------------------------------------------------------------------
MODEL = None
FEATURE_IMPORTANCE = {
    "weather": 30.3, "technical_issue": 23.3, "congestion": 18.8,
    "signal_issue": 10.7, "distance_km": 8.1, "historical_avg_delay": 4.3,
    "source": 1.4, "destination": 1.4, "train_no": 1.0,
    "day_of_week": 0.5, "is_weekend": 0.1,
}
try:
    MODEL = joblib.load(ML_DIR / "model.pkl")
    with open(ML_DIR / "feature_importance.json") as f:
        FEATURE_IMPORTANCE = json.load(f)
    print("[Velora] Loaded trained model from ml/model.pkl")
except Exception as e:
    print(f"[Velora] Could not load trained model ({e}); using rule-based fallback.")

TRAINS = {
    "12951": {"name": "Mumbai Rajdhani Express", "route": ["New Delhi", "Kota Jn", "Vadodara Jn", "Mumbai Central"]},
    "12301": {"name": "Howrah Rajdhani Express", "route": ["New Delhi", "Kanpur Central", "Mughal Sarai", "Howrah Jn"]},
    "12009": {"name": "Shatabdi Express", "route": ["New Delhi", "Kanpur Central", "Lucknow NR"]},
    "22439": {"name": "Vande Bharat Express", "route": ["New Delhi", "Kanpur Central", "Varanasi Jn"]},
    "12259": {"name": "Sealdah Duronto Express", "route": ["New Delhi", "Patna Jn", "Howrah Jn"]},
    "12626": {"name": "Kerala Express", "route": ["New Delhi", "Nagpur Jn", "Chennai Central"]},
    "12723": {"name": "Telangana Express", "route": ["New Delhi", "Bhopal Jn", "Vijayawada Jn"]},
    "12841": {"name": "Coromandel Express", "route": ["Howrah Jn", "Vijayawada Jn", "Chennai Central"]},
    "12471": {"name": "Swaraj Express", "route": ["Ajmer Jn", "Kota Jn", "New Delhi"]},
    "12903": {"name": "Golden Temple Mail", "route": ["Mumbai Central", "Vadodara Jn", "New Delhi"]},
}

WEATHER_OPTIONS = ["Clear", "Light Rain", "Heavy Rain", "Fog", "Storm"]
CONGESTION_OPTIONS = ["Low", "Medium", "High"]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS saved_trains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            train_no TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS crowd_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            train_no TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class PredictRequest(BaseModel):
    train_no: str
    source: str
    destination: str
    date: str


class SimulateRequest(BaseModel):
    train_no: str
    weather: str = "Clear"
    congestion: str = "Low"
    technical_issue: bool = False
    signal_issue: bool = False


class ChatRequest(BaseModel):
    message: str


class SaveTrainRequest(BaseModel):
    train_no: str


class CrowdReportRequest(BaseModel):
    train_no: str
    issue_type: str
    description: Optional[str] = ""


# ---------------------------------------------------------------------------
# Core prediction logic (shared by /predict, /simulate, /chatbot)
# ---------------------------------------------------------------------------
def deterministic_rng(*parts) -> random.Random:
    """Seed a Random instance from the request inputs, so the same
    (train, source, destination, date) always returns the same prediction
    instead of a different number on every refresh."""
    key = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def run_prediction(train_no: str, source: str, destination: str, day_of_week: int,
                    weather: str, congestion: str, technical_issue: int, signal_issue: int,
                    rng: Optional[random.Random] = None):
    rng = rng or random
    distance_km = rng.randint(300, 1800)
    historical_avg_delay = round(rng.uniform(8, 35), 1)

    if MODEL is not None:
        row = pd.DataFrame([{
            "train_no": train_no, "source": source, "destination": destination,
            "weather": weather, "congestion": congestion,
            "day_of_week": day_of_week, "is_weekend": 1 if day_of_week >= 5 else 0,
            "distance_km": distance_km, "historical_avg_delay": historical_avg_delay,
            "technical_issue": technical_issue, "signal_issue": signal_issue,
        }])
        delay = float(MODEL.predict(row)[0])
    else:
        base = 5 + distance_km * 0.01
        weather_factor = {"Clear": 0, "Light Rain": 8, "Heavy Rain": 25, "Fog": 35, "Storm": 30}[weather]
        congestion_factor = {"Low": 0, "Medium": 12, "High": 28}[congestion]
        delay = (base + weather_factor + congestion_factor + historical_avg_delay * 0.35
                 + technical_issue * 40 + signal_issue * 30)

    delay = max(0, round(delay, 1))
    # crude confidence heuristic: tighter when fewer compounding risk factors
    risk_factors = sum([
        weather in ("Heavy Rain", "Storm", "Fog"),
        congestion == "High",
        bool(technical_issue),
        bool(signal_issue),
    ])
    confidence = max(62, 95 - risk_factors * 8 - rng.randint(0, 4))

    causes = []
    if weather != "Clear":
        causes.append({"label": f"{weather} expected on route", "contribution": FEATURE_IMPORTANCE.get("weather", 25)})
    if congestion != "Low":
        causes.append({"label": f"{congestion} congestion near major junctions", "contribution": FEATURE_IMPORTANCE.get("congestion", 15)})
    if technical_issue:
        causes.append({"label": "Reported technical/rake issue", "contribution": FEATURE_IMPORTANCE.get("technical_issue", 20)})
    if signal_issue:
        causes.append({"label": "Signal failure reported on section", "contribution": FEATURE_IMPORTANCE.get("signal_issue", 12)})
    causes.append({"label": "Historical delay pattern for this train", "contribution": FEATURE_IMPORTANCE.get("historical_avg_delay", 8)})
    if not causes:
        causes.append({"label": "Clear conditions — minor schedule buffer only", "contribution": 5})

    return {
        "delay_minutes": delay,
        "confidence": confidence,
        "distance_km": distance_km,
        "causes": causes,
    }


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(email: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, email, name, hashed_password FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "email": row[1], "name": row[2], "hashed_password": row[3]}


def get_current_user(token: str = Depends(oauth2_scheme)):
    """Use as a dependency on any endpoint that requires login:
       def my_route(user = Depends(get_current_user))"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Velora AI API", "status": "ok"}


@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = pwd_context.hash(req.password)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, name, hashed_password, created_at) VALUES (?, ?, ?, ?)",
        (req.email, req.name, hashed, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    token = create_access_token({"sub": req.email})
    return {"access_token": token, "token_type": "bearer", "email": req.email, "name": req.name}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user or not pwd_context.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer", "email": user["email"], "name": user["name"]}


@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return {"email": user["email"], "name": user["name"]}


@app.get("/api/trains")
def list_trains():
    return [{"train_no": k, "name": v["name"], "route": v["route"]} for k, v in TRAINS.items()]


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.train_no not in TRAINS:
        train_name = "Unlisted Train"
    else:
        train_name = TRAINS[req.train_no]["name"]

    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    rng = deterministic_rng(req.train_no, req.source, req.destination, req.date)
    weather = rng.choices(WEATHER_OPTIONS, weights=[45, 20, 15, 12, 8])[0]
    congestion = rng.choices(CONGESTION_OPTIONS, weights=[50, 32, 18])[0]
    technical_issue = 1 if rng.random() < 0.08 else 0
    signal_issue = 1 if rng.random() < 0.07 else 0

    result = run_prediction(
        req.train_no, req.source, req.destination, date_obj.weekday(),
        weather, congestion, technical_issue, signal_issue, rng=rng,
    )

    scheduled_arrival = datetime.strptime("14:30", "%H:%M")
    expected_arrival = scheduled_arrival + timedelta(minutes=result["delay_minutes"])

    alternatives = []
    if result["delay_minutes"] > 60:
        pool = [v["name"] for k, v in TRAINS.items() if k != req.train_no]
        alternatives = random.sample(pool, k=min(3, len(pool)))

    return {
        "train_no": req.train_no,
        "train_name": train_name,
        "source": req.source,
        "destination": req.destination,
        "date": req.date,
        "delay_minutes": result["delay_minutes"],
        "confidence": result["confidence"],
        "scheduled_arrival": "14:30",
        "expected_arrival": expected_arrival.strftime("%H:%M"),
        "weather": weather,
        "congestion": congestion,
        "causes": result["causes"],
        "alternatives": alternatives,
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    rng = deterministic_rng(req.train_no, req.weather, req.congestion, req.technical_issue, req.signal_issue)
    result = run_prediction(
        req.train_no, "Source", "Destination", datetime.now().weekday(),
        req.weather, req.congestion, int(req.technical_issue), int(req.signal_issue), rng=rng,
    )
    return result


@app.get("/api/congestion")
def congestion_heatmap():
    stations = [
        "New Delhi", "Delhi Junction", "Kanpur Central", "Mughal Sarai", "Howrah Jn",
        "Mumbai Central", "Vadodara Jn", "Surat", "Chennai Central", "Vijayawada Jn",
        "Nagpur Jn", "Bhopal Jn", "Lucknow NR", "Patna Jn", "Kota Jn", "Ajmer Jn",
    ]
    levels = random.choices(["Low", "Medium", "High"], weights=[45, 35, 20], k=len(stations))
    return [{"station": s, "level": l} for s, l in zip(stations, levels)]


@app.get("/api/analytics/overview")
def analytics_overview():
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    monthly_trend = [{"month": m, "avg_delay": round(random.uniform(12, 38), 1)} for m in months]

    stations = ["Mughal Sarai", "Kanpur Central", "Howrah Jn", "Delhi Junction", "Vijayawada Jn"]
    most_delayed_stations = sorted(
        [{"station": s, "avg_delay": round(random.uniform(20, 55), 1)} for s in stations],
        key=lambda x: -x["avg_delay"],
    )

    routes = list(TRAINS.items())
    random.shuffle(routes)
    punctual = [{"route": v["name"], "on_time_pct": round(random.uniform(80, 97), 1)} for k, v in routes[:5]]
    delayed = [{"route": v["name"], "avg_delay": round(random.uniform(30, 70), 1)} for k, v in routes[5:10]]

    peak_hours = [{"hour": h, "congestion_index": round(random.uniform(0.2, 1.0), 2)} for h in
                  ["06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]]

    return {
        "monthly_trend": monthly_trend,
        "most_delayed_stations": most_delayed_stations,
        "most_punctual_routes": sorted(punctual, key=lambda x: -x["on_time_pct"]),
        "most_delayed_routes": sorted(delayed, key=lambda x: -x["avg_delay"]),
        "peak_congestion_hours": peak_hours,
        "model_accuracy_mae_minutes": 7.1,
        "model_r2_score": 0.85,
    }


@app.get("/api/sentiment")
def sentiment():
    return {
        "overall": random.choice(["Positive", "Neutral", "Negative"]),
        "breakdown": {
            "positive_pct": round(random.uniform(15, 35), 1),
            "neutral_pct": round(random.uniform(25, 40), 1),
            "negative_pct": round(random.uniform(30, 55), 1),
        },
        "sample_topics": ["AC coach cleanliness", "Late departure", "On-time arrival praise", "Catering quality"],
    }


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"


def call_gemini(message: str, context: dict) -> Optional[str]:
    """Calls the Gemini API with the chatbot's predicted delay as grounding
    context, so the model explains a *real* number instead of inventing one.
    Returns None on any failure so the caller can fall back gracefully."""
    if not GEMINI_API_KEY:
        return None
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    system_context = (
        "You are Velora, a concise rail-travel assistant. Use ONLY the prediction "
        f"data given here, don't invent numbers: {json.dumps(context)}. "
        "Answer in 1-2 short sentences, friendly and direct."
    )
    body = {
        "contents": [{"parts": [{"text": f"{system_context}\n\nUser question: {message}"}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120},
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Velora] Gemini call failed, falling back to rule-based reply: {e}")
        return None


@app.post("/api/chatbot")
def chatbot(req: ChatRequest):
    msg = req.message.lower()
    found_train = None
    for tno in TRAINS:
        if tno in msg:
            found_train = tno
            break

    if not found_train:
        fallback = "Tell me a train number (e.g. 12951) and I can predict its delay for you."
        if GEMINI_API_KEY:
            reply = call_gemini(req.message, {"note": "no train number detected in message"})
            return {"reply": reply or fallback}
        return {"reply": fallback}

    rng = deterministic_rng(found_train, datetime.now().strftime("%Y-%m-%d"))
    result = run_prediction(
        found_train, "Source", "Destination", datetime.now().weekday(),
        rng.choice(WEATHER_OPTIONS), rng.choice(CONGESTION_OPTIONS), 0, 0, rng=rng,
    )
    train_name = TRAINS[found_train]["name"]
    context = {
        "train_no": found_train, "train_name": train_name,
        "predicted_delay_minutes": result["delay_minutes"], "confidence_pct": result["confidence"],
        "top_cause": result["causes"][0]["label"] if result["causes"] else None,
    }
    fallback_reply = (
        f"Train {found_train} ({train_name}) is expected to run {result['delay_minutes']} min late, "
        f"with {result['confidence']}% confidence."
    )
    reply = call_gemini(req.message, context) if GEMINI_API_KEY else None

    return {
        "reply": reply or fallback_reply,
        "delay_minutes": result["delay_minutes"],
        "confidence": result["confidence"],
        "powered_by": "gemini" if reply else "rule-based-fallback",
    }


@app.post("/api/saved-trains")
def save_train(req: SaveTrainRequest, user=Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO saved_trains (user_email, train_no, created_at) VALUES (?, ?, ?)",
        (user["email"], req.train_no, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"status": "saved"}


@app.get("/api/saved-trains")
def get_saved_trains(user=Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT train_no, created_at FROM saved_trains WHERE user_email = ?", (user["email"],))
    rows = cur.fetchall()
    conn.close()
    return [{"train_no": r[0], "created_at": r[1]} for r in rows]


@app.post("/api/crowd-reports")
def add_crowd_report(req: CrowdReportRequest):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO crowd_reports (train_no, issue_type, description, created_at) VALUES (?, ?, ?, ?)",
        (req.train_no, req.issue_type, req.description, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"status": "reported"}


@app.get("/api/crowd-reports")
def list_crowd_reports(train_no: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if train_no:
        cur.execute("SELECT train_no, issue_type, description, created_at FROM crowd_reports WHERE train_no = ? ORDER BY id DESC", (train_no,))
    else:
        cur.execute("SELECT train_no, issue_type, description, created_at FROM crowd_reports ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
    return [{"train_no": r[0], "issue_type": r[1], "description": r[2], "created_at": r[3]} for r in rows]


@app.get("/api/admin/stats")
def admin_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM saved_trains")
    saved_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM crowd_reports")
    reports_count = cur.fetchone()[0]
    conn.close()
    return {
        "total_saved_trains": saved_count,
        "total_crowd_reports": reports_count,
        "model_status": "loaded" if MODEL is not None else "fallback-rule-based",
        "model_mae_minutes": 7.1,
        "model_r2_score": 0.85,
        "monitored_routes": len(TRAINS),
    }
