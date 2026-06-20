# Deploying Velora

Two pieces to deploy: the **backend** (FastAPI, needs a real server) and the
**frontend** (a static HTML file, needs only a CDN/static host).

## Backend — Render (free tier, easiest)

1. Push this repo to GitHub.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Render auto-detects `render.yaml` at the repo root. If it doesn't,
   set manually:
   - Root directory: `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (Render dashboard → Environment):
   - `VELORA_SECRET_KEY` — generate with
     `python -c "import secrets; print(secrets.token_hex(32))"`
   - `GEMINI_API_KEY` — from https://aistudio.google.com/app/apikey
     (optional — chatbot falls back to rule-based replies without it)
5. Deploy. You'll get a URL like `https://velora-backend.onrender.com`.

> Free-tier Render services sleep after inactivity and take ~30s to wake up
> on the first request — mention this if a recruiter tries the live link
> cold.

## Backend — Railway (alternative)

1. https://railway.app → New Project → Deploy from GitHub repo.
2. Railway reads the `Procfile` at the repo root automatically.
3. Add the same two environment variables as above in Railway's Variables tab.
4. Railway gives you a public URL once deployed.

## Frontend — Vercel or Netlify (static, free)

The frontend is a single `index.html` with no build step.

**Vercel:**
1. https://vercel.com → New Project → import the repo.
2. Set root directory to `frontend`.
3. Framework preset: "Other" (no build command needed).
4. Deploy.

**Netlify (drag-and-drop, fastest):**
1. https://app.netlify.com/drop
2. Drag the `frontend` folder onto the page. Done — you get a live URL
   immediately, no GitHub needed.

## Connecting them

Open `frontend/index.html` and change this line near the top of the
`<script>` block:

```js
const API_BASE = (window.location.protocol === 'file:') ? null : 'http://localhost:8000';
```

to your deployed backend URL:

```js
const API_BASE = 'https://velora-backend.onrender.com';
```

Then re-deploy the frontend (or just re-upload to Netlify drop).

## CORS note

`app.py` currently allows all origins (`allow_origins=["*"]`) for demo
simplicity. Before calling this "production," lock it down to your actual
frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Put on your resume

Once both are live, add the link as:
`Live demo: https://your-frontend-url.vercel.app (backend: Render)`

That single live link is worth more in an interview than any amount of
"works on my machine" — it's proof you can ship, not just train a model.
