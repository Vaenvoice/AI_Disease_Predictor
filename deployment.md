# Deployment Guide: Vaen Health AI

To get a live link for this project, you need to deploy the **Frontend** and the **Backend** separately (or together on a platform like Render).

## 🚀 Recommended Deployment Strategy (Render)

Render is an excellent choice for full-stack applications with a Python backend and a React frontend.

### 1. Deploy the Backend (FastAPI)
- **Service Type**: Web Service
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app`
- **Environment Variables**: Set `ARTIFACTS_PATH` if needed, though absolute paths should work if the folder structure is preserved.

### 2. Deploy the Frontend (React/Vite)
- **Service Type**: Static Site
- **Build Command**: `npm install && npm run build`
- **Publish Directory**: `frontend/dist`
- **Environment Variables**: Set `VITE_API_BASE` to your Backend URL.

##  alternativas (Vercel + Railway)
- **Frontend**: Deploy to **Vercel** (connect your GitHub repo, select the `frontend` directory).
- **Backend**: Deploy to **Railway** or **Render** for the Python API.

---

### Important Considerations
- **CORS**: Ensure the Backend has the Frontend's live URL in its CORS `allow_origins` list.
- **Model Artifacts**: Ensure the `.pkl` files in `models/` are pushed to GitHub so the live server can load them.
