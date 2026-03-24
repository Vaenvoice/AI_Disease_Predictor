# Vaen Health AI: Disease Predictor

Vaen Health AI is a professional, full-stack medical diagnostic platform that uses machine learning to evaluate clinical parameters and predict the risk of chronic diseases. 

## 🚀 Features

- **Multi-Disease Analysis**: Specialized modules for Diabetes, Heart Disease, Kidney Disease, and Parkinson's.
- **Enterprise UI**: A minimalist, high-fidelity interface inspired by Google's health initiatives, built with React and Vite.
- **Research-Backed Models**: Powered by optimized Random Forest and SVM models with high accuracy (over 95% for Kidney Disease).
- **Clinical Interpretability**: Real-time tooltips providing medical context for complex clinical parameters.
- **Privacy-Centric**: Fast, locally-driven inferences using a robust FastAPI backend.

## 🛠️ Architecture

- **Frontend**: React (Vite) + CSS (Premium Minimalist Design)
- **Backend**: FastAPI (Python)
- **ML Models**: Scikit-learn (Random Forest, SVM)
- **Persistence**: Joblib Artifacts

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 16+

### Backend Setup
1. Navigate to `backend/`
2. Install dependencies: `pip install -r requirements.txt` (Ensure `pandas`, `joblib`, `scikit-learn`, `fastapi`, and `uvicorn` are installed)
3. Start the server: `python main.py` (Runs on http://localhost:8000)

### Frontend Setup
1. Navigate to `frontend/`
2. Install dependencies: `npm install`
3. Start the dev server: `npm run dev` (Runs on http://localhost:5173)

## 📸 UI Showcase
Refer to the `reports/` directory for visual previews and model performance metrics.

## ⚖️ License
MIT License - Created for educational and clinical research purposes by the Vaen Health team.
