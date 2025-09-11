import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ========================
# 🎨 Theme Handler
# ========================
def apply_theme():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    choice = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

    if choice == "Dark":
        st.session_state.dark_mode = True
        st.markdown(
            """
            <style>
            body, .stApp {
                background-color: #121212 !important;
                color: #ffffff !important;
            }
            .stSidebar, .css-1d391kg, .css-qrbaxs {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            .stButton>button {
                background-color: #bb86fc !important;
                color: black !important;
                border-radius: 8px !important;
                font-weight: bold;
            }
            .stNumberInput>div>input, .stSelectbox>div>select, .stSlider>div>div>input {
                background-color: #333 !important;
                color: #ffffff !important;
                font-weight: bold;
            }
            h1, h2, h3, h4, h5, h6, label, .css-10trblm {
                color: #ffffff !important;
                text-shadow: 0px 0px 6px rgba(255,255,255,0.8);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state.dark_mode = False
        st.markdown(
            """
            <style>
            body, .stApp {
                background-color: white !important;
                color: black !important;
            }
            .stButton>button {
                background-color: #007bff !important;
                color: white !important;
                border-radius: 8px !important;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# ========================
# 🛠 Feature Alignment
# ========================
def prepare_input(input_dict, feature_file, scaler_file):
    features = joblib.load(feature_file)
    scaler = joblib.load(scaler_file)

    df_input = pd.DataFrame([input_dict])

    # Add missing columns
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Correct order
    df_input = df_input[features]

    return scaler.transform(df_input)

# ========================
# 📊 Prediction Display
# ========================
def show_prediction(prediction, probabilities, positive_label, negative_label):
    prob_positive = probabilities[1]
    prob_negative = probabilities[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{positive_label}")
        st.progress(prob_positive)
        st.write(f"{prob_positive*100:.1f}%")
    with col2:
        st.markdown(f"{negative_label}")
        st.progress(prob_negative)
        st.write(f"{prob_negative*100:.1f}%")

    if prediction == 1:
        st.markdown(
            f"<h3 style='color:#dc3545;'>Prediction: {positive_label}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h3 style='color:#28a745;'>Prediction: {negative_label}</h3>",
            unsafe_allow_html=True,
        )

# ========================
# 🚀 Main App
# ========================
apply_theme()

with st.sidebar:
    st.title("🧠 AI Disease Predictor")
    st.write("Choose a disease and enter your health data to get predictions.")
    st.write("Developed with ❤ using Streamlit")

disease = st.selectbox("Choose a Disease", ["Diabetes", "Heart Disease", "Parkinson's"])

# ---------------- Diabetes ----------------
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    pregnancies = st.slider("Pregnancies (0–20)", 0, 20, 0, help="Number of times pregnant")
    glucose = st.slider("Glucose (40–200)", 40, 200, 120, help="Fasting blood sugar level. Ideal: <100 mg/dL")
    blood_pressure = st.slider("Blood Pressure (40–140)", 40, 140, 70, help="Diastolic BP. Normal: <80 mmHg")
    skin_thickness = st.slider("Skin Thickness (5–60)", 5, 60, 20, help="Indicator of body fat")
    insulin = st.slider("Insulin (15–276)", 15, 276, 80, help="Insulin level. Normal: 16–166 μU/mL")
    bmi = st.slider("BMI (10–50)", 10.0, 50.0, 25.0, help="Healthy BMI: 18.5–24.9")
    diabetes_pedigree = st.slider("Diabetes Pedigree Function (0–2.5)", 0.0, 2.5, 0.5, help="Likelihood from family history")
    age = st.slider("Age (20–80)", 20, 80, 30, help="Older age increases risk")

    if st.button("Predict Diabetes"):
        try:
            model = joblib.load("models/diabetes.pkl")
            input_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "Age": age
            }
            input_scaled = prepare_input(input_data, "models/diabetes_features.pkl", "models/diabetes_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Diabetes Detected", "No Diabetes")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ---------------- Heart Disease ----------------
elif disease == "Heart Disease":
    st.header("❤ Heart Disease Prediction")

    age = st.slider("Age", 20, 80, 45, help="Older age increases risk")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Males usually higher risk")
    cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1, help="0=Typical Angina, 3=Asymptomatic")
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120, help="Normal: <120 mmHg")
    chol = st.slider("Cholesterol", 100, 600, 240, help="Normal: <200 mg/dL")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = Yes (possible diabetes)")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], help="0=Normal")
    thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150, help="Higher capacity = healthier")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], help="1 = Yes")
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, help="Exercise-induced depression")
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2], help="0=Upsloping, 2=Downsloping (riskier)")
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3], help="Higher = more blockage risk")
    thal = st.selectbox("Thal (0=Normal, 1=Fixed, 2=Reversible)", [0, 1, 2])

    if st.button("Predict Heart Disease"):
        try:
            model = joblib.load("models/heart.pkl")
            input_data = {
                "age": age,
                "sex": 1 if sex == "Male" else 0,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
            input_scaled = prepare_input(input_data, "models/heart_features.pkl", "models/heart_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Heart Disease Detected", "No Heart Disease")
        except Exception as e:
            st.error(f"⚠ Error: {e}")

# ---------------- Parkinson's ----------------
elif disease == "Parkinson's":
    st.header("🧩 Parkinson's Prediction")

    fo = st.slider("Average Vocal Fundamental Frequency (fo)", 100.0, 400.0, 150.0, help="Avg voice pitch")
    fhi = st.slider("Maximum Vocal Frequency (fhi)", 150.0, 500.0, 200.0, help="Max pitch")
    flo = st.slider("Minimum Vocal Frequency (flo)", 50.0, 200.0, 100.0, help="Min pitch")
    jitter = st.slider("Jitter (%)", 0.0, 1.0, 0.01, 0.001, help="Voice frequency variation")
    shimmer = st.slider("Shimmer", 0.0, 1.0, 0.02, 0.001, help="Voice amplitude variation")
    rpde = st.slider("RPDE", 0.0, 1.0, 0.5, 0.01, help="Voice disorder measure")
    dfa = st.slider("DFA", 0.0, 1.0, 0.5, 0.01, help="Signal fractal scaling")
    spread1 = st.slider("Spread1", -10.0, 0.0, -5.0, 0.1, help="Voice frequency spread")
    spread2 = st.slider("Spread2", -5.0, 5.0, 0.0, 0.1, help="Voice stability")
    d2 = st.slider("D2", 1.0, 5.0, 2.0, 0.1, help="Signal complexity")
    PPE = st.slider("PPE", 0.0, 1.0, 0.1, 0.01, help="Pitch Period Entropy")

    if st.button("Predict Parkinson's"):
        try:
            model = joblib.load("models/parkinsons.pkl")
            input_data = {
                "MDVP:Fo(Hz)": fo,
                "MDVP:Fhi(Hz)": fhi,
                "MDVP:Flo(Hz)": flo,
                "MDVP:Jitter(%)": jitter,
                "MDVP:Shimmer": shimmer,
                "RPDE": rpde,
                "DFA": dfa,
                "spread1": spread1,
                "spread2": spread2,
                "D2": d2,
                "PPE": PPE
            }
            input_scaled = prepare_input(input_data, "models/parkinsons_features.pkl", "models/parkinsons_scaler.pkl")
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            show_prediction(prediction, probabilities, "Parkinson's Detected", "No Parkinson's")
        except Exception as e:
            st.error(f"⚠ Error: {e}")
