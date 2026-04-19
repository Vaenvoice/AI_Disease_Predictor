# Deployment Guide: Vaen Health AI (Manual Render Setup)

This guide provides instructions for deploying a **new Backend** and connecting your **existing Frontend** on Render.

---

## 🚀 1. Deploy the New Backend

Since your previous backend account was suspended, follow these steps to deploy on a new Render account:

1.  Log in to your **new Render account**.
2.  Click **"New"** → **"Web Service"**.
3.  Connect this GitHub repository.
4.  Configure the service with these settings:
    -   **Name**: `ai-disease-predictor-backend`
    -   **Region**: (Choose the one closest to you)
    -   **Language**: `Python 3`
    -   **Build Command**: `pip install -r backend/requirements.txt`
    -   **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:$PORT`
    -   **Instance Type**: `Free`
5.  Click **"Create Web Service"**.
6.  Once deployed, copy the **Public URL** (e.g., `https://ai-disease-predictor-backend.onrender.com`).

---

## 🌐 2. Connect Your Existing Frontend

Your frontend is already live at `https://vaen-health.onrender.com`. You just need to update it to point to the new backend:

1.  Go to your **Frontend Static Site** dashboard on Render.
2.  Go to **"Settings"** → **"Environment Variables"**.
3.  Update (or add) the following variable:
    -   **Key**: `VITE_API_BASE`
    -   **Value**: `https://your-new-backend-url.onrender.com` (Paste your new backend URL here)
4.  Render will automatically re-deploy your frontend with the new connection.

---

## 🛠️ 3. Prevent Backend Sleep (Keep-Alive)

To prevent the Free Tier backend from spinning down after 15 minutes of inactivity:

1.  Go to your **GitHub Repository** → **Settings** → **Secrets and variables** → **Actions**.
2.  Update the following secrets:
    -   **`RENDER_BACKEND_URL`**: `https://your-new-backend-url.onrender.com/health`
    -   **`RENDER_FRONTEND_URL`**: `https://vaen-health.onrender.com`
3.  Ensure the **"Keep Alive"** workflow is enabled in the **Actions** tab of your repository.

---

## ⚙️ Summary of Local Changes
-   **Cleanup**: Removed `render.yaml`, `railway.toml`, and other legacy configs.
-   **Heartbeat**: Added an internal heartbeat to `App.jsx` that pings the backend every 10 minutes while the site is open in a browser.
-   **Portability**: The GitHub Action now uses secrets so you can switch accounts easily without editing code.
