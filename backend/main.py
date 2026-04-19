# import pandas as pd # Removed for memory optimization
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI_Disease_Predictor")

app = FastAPI(title="AI Disease Predictor API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS & SCALERS CACHE ---
MODELS = {}
SCALERS = {}
DISEASES = ["diabetes", "heart", "kidney", "parkinsons"]

# Models live at backend/models/ — always relative to this file, no guessing needed.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
logger.info(f"Models directory: {_MODELS_DIR}")
logger.info(f"Models directory exists: {os.path.isdir(_MODELS_DIR)}")
if os.path.isdir(_MODELS_DIR):
    logger.info(f"Models directory contents: {os.listdir(_MODELS_DIR)}")

def load_artifacts(disease):
    """Load model and scaler for a specific disease."""
    if disease not in MODELS:
        model_path = os.path.join(_MODELS_DIR, f"{disease}_best_model.pkl")
        scaler_path = os.path.join(_MODELS_DIR, f"{disease}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Artifacts for {disease} not found at {model_path}.")

        logger.info(f"Loading artifacts for {disease}...")
        MODELS[disease] = joblib.load(model_path)
        SCALERS[disease] = joblib.load(scaler_path)
    return MODELS[disease], SCALERS[disease]

# --- STARTUP EVENT REMOVED FOR LAZY LOADING ---
# Pre-loading models on startup exceeds memory/time limits on Free Tier.
# Models will now load on-demand during the first request.

# --- REQUEST SCHEMAS ---
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class KidneyInput(BaseModel):
    bp_diastolic: float
    bp_limit: float
    sg: float
    al: float
    rbc: float
    su: float
    pc: float
    pcc: float
    ba: float
    bgr: float
    bu: float
    sod: float
    sc: float
    pot: float
    hemo: float
    pcv: float
    rbcc: float
    wbcc: float
    htn: float
    dm: float
    cad: float
    appet: float
    pe: float
    ane: float
    grf: float
    stage: float
    affected: float
    age: float

class ParkinsonsInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    jitter_percent: float
    jitter_abs: float
    rap: float
    ppq: float
    ddp: float
    shimmer: float
    shimmer_db: float
    apq3: float
    apq5: float
    apq: float
    dda: float
    nhr: float
    hnr: float
    rpde: float
    dfa: float
    spread1: float
    spread2: float
    d2: float
    ppe: float

# --- ENDPOINTS ---
@app.api_route("/", methods=["GET", "HEAD"])
async def home():
    """Home endpoint that confirms the API is running (supports GET and HEAD for uptime monitoring)."""
    return {"message": "AI Disease Predictor API is running."}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    """Structured health check endpoint for monitoring services like Uptime Robot."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "service": "AI Disease Predictor API"
    }

@app.post("/predict/diabetes")
async def predict_diabetes(data: DiabetesInput):
    start_time = time.time()
    try:
        logger.info("Processing diabetes prediction request...")
        model, scaler = load_artifacts("diabetes")
        # Use NumPy for faster processing than Pandas for single-row inference
        input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])
        
        logger.debug("Scaling features...")
        scaled_features = scaler.transform(input_data)
        
        logger.debug("Running model inference...")
        prob = model.predict_proba(scaled_features)[0][1]
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Diabetes prediction completed in {latency:.2f}ms")
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5), "latency_ms": latency}
    except Exception as e:
        logger.error(f"Diabetes prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/heart")
async def predict_heart(data: HeartInput):
    start_time = time.time()
    try:
        logger.info("Processing heart prediction request...")
        model, scaler = load_artifacts("heart")
        input_data = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])
        
        logger.debug("Scaling features...")
        scaled_features = scaler.transform(input_data)
        
        logger.debug("Running model inference...")
        prob = model.predict_proba(scaled_features)[0][1]
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Heart prediction completed in {latency:.2f}ms")
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5), "latency_ms": latency}
    except Exception as e:
        logger.error(f"Heart prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/kidney")
async def predict_kidney(data: KidneyInput):
    start_time = time.time()
    try:
        logger.info("Processing kidney prediction request...")
        model, scaler = load_artifacts("kidney")
        input_data = np.array([[data.bp_diastolic, data.bp_limit, data.sg, data.al, data.rbc, data.su, data.pc, data.pcc, data.ba, data.bgr, data.bu, data.sod, data.sc, data.pot, data.hemo, data.pcv, data.rbcc, data.wbcc, data.htn, data.dm, data.cad, data.appet, data.pe, data.ane, data.grf, data.stage, data.affected, data.age]])
        
        logger.debug("Scaling features...")
        scaled_features = scaler.transform(input_data)
        
        logger.debug("Running model inference...")
        try:
            prob = model.predict_proba(scaled_features)[0][1]
        except Exception:
            logger.warning("predict_proba failed, falling back to predict")
            prediction = model.predict(scaled_features)[0]
            prob = 1.0 if prediction == 1 else 0.0
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Kidney prediction completed in {latency:.2f}ms")
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5), "latency_ms": latency}
    except Exception as e:
        logger.error(f"Kidney prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/parkinsons")
async def predict_parkinsons(data: ParkinsonsInput):
    start_time = time.time()
    try:
        logger.info("Processing parkinsons prediction request...")
        model, scaler = load_artifacts("parkinsons")
        input_data = np.array([[data.fo, data.fhi, data.flo, data.jitter_percent, data.jitter_abs, data.rap, data.ppq, data.ddp, data.shimmer, data.shimmer_db, data.apq3, data.apq5, data.apq, data.dda, data.nhr, data.hnr, data.rpde, data.dfa, data.spread1, data.spread2, data.d2, data.ppe]])
        
        logger.debug("Scaling features...")
        scaled_features = scaler.transform(input_data)
        
        logger.debug("Running model inference...")
        prob = model.predict_proba(scaled_features)[0][1]
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Parkinson's prediction completed in {latency:.2f}ms")
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5), "latency_ms": latency}
    except Exception as e:
        logger.error(f"Parkinson's prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
