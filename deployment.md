# Deployment Guide: Vaen Health AI (Render)

This guide explains how to deploy the AI Disease Predictor to **Render** and keep it from "sleeping" on the free tier.

---

## 🚀 Easy Deployment (Blueprint)

The repository includes a `render.yaml` file that allows you to deploy both the Backend and Frontend in one go.

1.  Log in to [Render](https://render.com).
2.  Click **"New"** → **"Blueprint"**.
3.  Connect this GitHub repository.
4.  Render will detect `render.yaml` and offer to create the following services:
    -   `ai-disease-predictor-backend`
    -   `ai-disease-predictor-frontend`
5.  Click **"Apply"**.

---

## 🛠️ Configuration & Secrets

### 1. Frontend Backend URL
The `render.yaml` automatically connects the frontend to the backend. If you need to change it manually:
-   In the Frontend service settings, set the environment variable:
    -   `VITE_API_BASE` = `https://your-backend-url.onrender.com`

### 2. Preventing Sleep (Keep-Alive)
Render's free tier spins down after 15 minutes of inactivity. We solve this using a **GitHub Action** that pings your app every 10 minutes.

**Setup Steps:**
1.  Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**.
2.  Add a **New repository secret**:
    -   **Name**: `RENDER_BACKEND_URL`
    -   **Value**: `https://your-backend-url.onrender.com/health`
3.  Add another secret (optional):
    -   **Name**: `RENDER_FRONTEND_URL`
    -   **Value**: `https://your-frontend-url.onrender.com`
4.  Go to the **Actions** tab in GitHub and ensure the "Keep Alive" workflow is enabled.

---

## ⚙️ Backend Details
-   **Runtime**: Python 3.10
-   **Build Command**: `pip install -r backend/requirements.txt`
-   **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:$PORT`
-   **Health Check**: `/health`

## 🌐 Frontend Details
-   **Runtime**: Static Site
-   **Build Command**: `cd frontend && npm install && npm run build`
-   **Publish Directory**: `frontend/dist`

---

## 🧪 Verification
After deployment, open your browser's Developer Tools (F12) and check the **Network** tab. You should see a "pulse" request every 10 minutes to the `/health` endpoint while the tab is open. This is the **Frontend Heartbeat** which keeps the backend warm while you browse.
