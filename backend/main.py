import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

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

def load_artifacts(disease):
    if disease not in MODELS:
        # Resolve path relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", f"{disease}_best_model.pkl")
        scaler_path = os.path.join(base_dir, "models", f"{disease}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Artifacts for {disease} not found at {model_path}.")
            
        MODELS[disease] = joblib.load(model_path)
        SCALERS[disease] = joblib.load(scaler_path)
    return MODELS[disease], SCALERS[disease]

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
@app.get("/")
def home():
    return {"message": "AI Disease Predictor API is running."}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        model, scaler = load_artifacts("diabetes")
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        df = pd.DataFrame([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]], columns=feature_names)
        scaled_features = scaler.transform(df)
        prob = model.predict_proba(scaled_features)[0][1]
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        model, scaler = load_artifacts("heart")
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df = pd.DataFrame([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]], columns=feature_names)
        scaled_features = scaler.transform(df)
        prob = model.predict_proba(scaled_features)[0][1]
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/kidney")
def predict_kidney(data: KidneyInput):
    try:
        model, scaler = load_artifacts("kidney")
        feature_names = ['bp (Diastolic)', 'bp limit', 'sg', 'al', 'rbc', 'su', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'stage', 'affected', 'age']
        df = pd.DataFrame([[data.bp_diastolic, data.bp_limit, data.sg, data.al, data.rbc, data.su, data.pc, data.pcc, data.ba, data.bgr, data.bu, data.sod, data.sc, data.pot, data.hemo, data.pcv, data.rbcc, data.wbcc, data.htn, data.dm, data.cad, data.appet, data.pe, data.ane, data.grf, data.stage, data.affected, data.age]], columns=feature_names)
        scaled_features = scaler.transform(df)
        try:
            prob = model.predict_proba(scaled_features)[0][1]
        except:
            prediction = model.predict(scaled_features)[0]
            prob = 1.0 if prediction == 1 else 0.0
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/parkinsons")
def predict_parkinsons(data: ParkinsonsInput):
    try:
        model, scaler = load_artifacts("parkinsons")
        feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        df = pd.DataFrame([[data.fo, data.fhi, data.flo, data.jitter_percent, data.jitter_abs, data.rap, data.ppq, data.ddp, data.shimmer, data.shimmer_db, data.apq3, data.apq5, data.apq, data.dda, data.nhr, data.hnr, data.rpde, data.dfa, data.spread1, data.spread2, data.d2, data.ppe]], columns=feature_names)
        scaled_features = scaler.transform(df)
        prob = model.predict_proba(scaled_features)[0][1]
        return {"risk_probability": float(prob), "prediction": int(prob > 0.5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
