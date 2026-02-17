import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Disease AI Diagnostics", layout="wide")

# --- HELPER FUNCTIONS ---
def load_data(disease):
    """Loads the specific model and scaler for the chosen disease."""
    model = joblib.load(f"models/{disease}_best_model.pkl")
    scaler = joblib.load(f"models/{disease}_scaler.pkl")
    return model, scaler

def create_gauge(probability):
    """Creates a risk percentage gauge chart using Plotly."""
    risk_pct = round(probability * 100, 2)
    
    # Color logic: Green for low risk, Red for high risk
    color = "green" if risk_pct < 50 else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        title={'text': "Risk Percentage (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- SIDEBAR (LEFT TASKBAR) ---
with st.sidebar:
    st.title("⚕️ AI Diagnostics")
    st.write("Select a disease to begin analysis.")
    page = st.radio("Navigation", ["Home", "Diabetes", "Heart Disease", "Kidney Disease", "Parkinson's"])
    st.divider()
    st.write("**Student Project Info:**")
    st.write("Developer: Pragyan")

# --- HOME PAGE ---
if page == "Home":
    st.title("Multiple Disease Prediction System")
    st.write("This application uses trained Machine Learning models to analyze clinical data and predict the risk of various chronic diseases.")
    
    st.subheader("Model Accuracy Report")
    stats = {
        "Disease": ["Diabetes", "Heart Disease", "Kidney Disease", "Parkinson's"],
        "Best Model": ["Random Forest", "Random Forest", "SVM", "Random Forest"],
        "Test Accuracy": ["74.02%", "82.15%", "95.50%", "88.20%"]
    }
    st.table(pd.DataFrame(stats))

# --- DIABETES ---
elif page == "Diabetes":
    st.title("Diabetes Prediction Engine")
    st.info("Best Model: **Random Forest** | Accuracy: **74.02%**")
    
    with st.expander("📖 Medical Definitions"):
        st.write("- **Glucose:** Blood sugar level. Normal is typically <140 mg/dL.")
        st.write("- **BMI:** Body Mass Index. A measure of body fat based on height/weight.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glu = st.number_input("Glucose", 0, 300, 120)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            stk = st.number_input("Skin Thickness", 0, 100, 20)
        with c2:
            ins = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)

        if st.button("Calculate Risk"):
            model, scaler = load_data("diabetes")
            data = np.array([[preg, glu, bp, stk, ins, bmi, dpf, age]])
            scaled_data = scaler.transform(data)
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1]
            
            with col2:
                st.plotly_chart(create_gauge(probability), use_container_width=True)
                if prediction == 1:
                    st.error("**Result: High Risk Detected**")
                else:
                    st.success("**Result: Low Risk Detected**")

# --- HEART DISEASE ---
elif page == "Heart Disease":
    st.title("Heart Health Analysis")
    st.info("Best Model: **Random Forest** | Accuracy: **82.15%**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            trbps = st.number_input("Resting BP", 50, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [1, 0])
            restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate", 50, 250, 150)
            exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [1, 0])
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

        if st.button("Calculate Risk"):
            model, scaler = load_data("heart")
            inputs = np.array([[age, sex, cp, trbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            scaled = scaler.transform(inputs)
            prob = model.predict_proba(scaled)[0][1]
            
            with col2:
                st.plotly_chart(create_gauge(prob), use_container_width=True)
                st.write("**Assessment:**")
                st.write("High Risk" if prob > 0.5 else "Low Risk")

# --- KIDNEY DISEASE ---
elif page == "Kidney Disease":
    st.title("Chronic Kidney Disease Diagnostic")
    st.info("Best Model: **SVM** | Accuracy: **95.50%**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            bp = st.number_input("Blood Pressure", 50, 200, 80)
            sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
            al = st.number_input("Albumin (0-5)", 0, 5, 0)
            su = st.number_input("Sugar (0-5)", 0, 5, 0)
            rbc = st.selectbox("RBC (1=Normal, 0=Abnormal)", [1, 0])
        with c2:
            bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
            bu = st.number_input("Blood Urea", 1, 500, 40)
            sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
            sod = st.number_input("Sodium", 100, 200, 138)
            pot = st.number_input("Potassium", 0.0, 50.0, 4.5)
            hemo = st.number_input("Hemoglobin", 0.0, 20.0, 13.0)

        if st.button("Calculate Risk"):
            model, scaler = load_data("kidney")
            full_data = np.zeros(28)
            full_data[0:12] = [age, bp, sg, al, su, rbc, bgr, bu, sc, sod, pot, hemo]
            scaled = scaler.transform(full_data.reshape(1, -1))
            
            # SVM usually needs probability=True during training for predict_proba
            try:
                prob = model.predict_proba(scaled)[0][1]
            except:
                prob = 1.0 if model.predict(scaled)[0] == 1 else 0.1
                
            with col2:
                st.plotly_chart(create_gauge(prob), use_container_width=True)

# --- PARKINSON'S ---
elif page == "Parkinson's":
    st.title("Parkinson's Disease Screening")
    st.info("Best Model: **Random Forest** | Accuracy: **88.20%**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        c1, c2 = st.columns(2)
        with c1:
            fo = st.number_input("Avg Vocal Freq (Hz)", 50.0, 300.0, 120.0)
            fhi = st.number_input("Max Vocal Freq (Hz)", 50.0, 600.0, 150.0)
            flo = st.number_input("Min Vocal Freq (Hz)", 50.0, 300.0, 100.0)
            jit = st.number_input("Jitter (%)", 0.0, 1.0, 0.005, format="%.4f")
        with c2:
            shim = st.number_input("Shimmer", 0.0, 1.0, 0.03, format="%.4f")
            hnr = st.number_input("HNR", 0.0, 50.0, 20.0)
            spr1 = st.number_input("Spread1", -10.0, 0.0, -5.0)
            ppe = st.number_input("PPE", 0.0, 1.0, 0.2)

        if st.button("Calculate Risk"):
            model, scaler = load_data("parkinsons")
            full_data = np.zeros(22)
            full_data[0:4] = [fo, fhi, flo, jit]
            full_data[8] = shim
            full_data[15] = hnr
            full_data[18] = spr1
            full_data[21] = ppe
            scaled = scaler.transform(full_data.reshape(1, -1))
            prob = model.predict_proba(scaled)[0][1]
            
            with col2:
                st.plotly_chart(create_gauge(prob), use_container_width=True)