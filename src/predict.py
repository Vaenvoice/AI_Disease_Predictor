import numpy as np
import joblib


def load_artifacts():
    model = joblib.load("models/diabetes_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def predict_diabetes(input_data):

    model, scaler = load_artifacts()

    # Convert to numpy array
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability


if __name__ == "__main__":

    print("Enter Patient Details:")

    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    input_data = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]

    prediction, probability = predict_diabetes(input_data)

    if prediction == 1:
        print(f"\n⚠ High Risk of Diabetes")
    else:
        print(f"\n✅ Low Risk of Diabetes")

    print(f"Risk Probability: {probability:.2f}")
