# Deployment Guide: Vaen Health AI

The **Frontend** is deployed separately from the **Backend**.

---

## 🚀 Backend — Railway (FastAPI)

Railway is the primary hosting platform for the Python backend.

### Setup Steps

1. Go to [railway.app](https://railway.app) and create a new project.
2. Click **"Deploy from GitHub repo"** and connect this repository.
3. Railway will auto-detect the `railway.toml` and `nixpacks.toml` and configure the build.
4. Once deployed, copy the **public URL** Railway generates (e.g. `https://your-app.up.railway.app`).

### Build & Start Commands (auto-configured via `railway.toml`)
- **Build**: `pip install -r backend/requirements.txt`
- **Start**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:$PORT`
- **Health Check**: `/health`

### Environment Variables (set in Railway dashboard)
No required secrets — models are loaded from the `models/` directory committed to git.

> ⚠️ **Important**: Make sure all `.pkl` files in `models/` are pushed to GitHub so Railway can load them at startup.

---

## 🌐 Frontend — Vercel / Render Static Site (React/Vite)

### Vercel (Recommended)
1. Go to [vercel.com](https://vercel.com) and import this GitHub repo.
2. Set **Root Directory** to `frontend`.
3. Set the environment variable:
   - `VITE_API_BASE` = `https://your-app.up.railway.app` ← paste your Railway backend URL here
4. Deploy.

### Render Static Site (Alternative)
- **Build Command**: `cd frontend && npm install && npm run build`
- **Publish Directory**: `frontend/dist`
- **Environment Variable**: `VITE_API_BASE` = `https://your-app.up.railway.app`

---

## ⚙️ CORS

The backend uses `allow_origins=["*"]` so all frontend origins are accepted by default.
To restrict to your specific frontend URL, update `main.py`:

```python
allow_origins=["https://vaen-health.vercel.app"]
```

---

## 🛠️ Uptime Monitoring

Railway's free tier (Hobby plan) **does not spin down** like Render — no keep-alive pings needed.
If you'd still like monitoring:
- **Service**: [Uptime Robot](https://uptimerobot.com/)
- **URL**: `https://your-app.up.railway.app/health`
- **Interval**: 5 minutes
