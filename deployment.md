# Deployment Guide: Vaen Health AI

To get a live link for this project, you need to deploy the **Frontend** and the **Backend** separately (or together on a platform like Render).

## 🚀 Recommended Deployment Strategy (Render)

Render is an excellent choice for full-stack applications with a Python backend and a React frontend.

### 1. Deploy the Backend (FastAPI)
- **Service Type**: Web Service
- **Link**: [https://ai-disease-predictor-nbdj.onrender.com](https://ai-disease-predictor-nbdj.onrender.com)
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app`
- **Root Directory**: `.` (Project Root)

### 2. Deploy the Frontend (React/Vite)
- **Service Type**: Static Site
- **Link**: [https://vaen-health.onrender.com](https://vaen-health.onrender.com)
- **Build Command**: `cd frontend && npm install && npm run build`
- **Publish Directory**: `frontend/dist`
- **Environment Variables**: Set `VITE_API_BASE` to `https://ai-disease-predictor-nbdj.onrender.com`.

##  alternativas (Vercel + Railway)
- **Frontend**: Deploy to **Vercel** (connect your GitHub repo, select the `frontend` directory).
- **Backend**: Deploy to **Railway** or **Render** for the Python API.

---

### Important Considerations
- **CORS**: Ensure the Backend has the Frontend's live URL in its CORS `allow_origins` list.
- **Model Artifacts**: Ensure the `.pkl` files in `models/` are pushed to GitHub so the live server can load them.
