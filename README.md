🧠 AI Disease Prediction System

An AI-powered web application that predicts the likelihood of diseases such as Diabetes, Heart Disease, and Parkinson's based on user-provided health parameters. Built using Python and Streamlit, this system leverages machine learning models to assist in early disease detection.

📌 Project Overview

This project aims to provide users with an interactive platform to assess their risk for certain diseases. By inputting specific health metrics, users receive predictions that can guide them toward seeking medical advice or making lifestyle changes.

🧪 Features

Disease Prediction: Predicts the likelihood of Diabetes, Heart Disease, and Parkinson's based on user inputs.

Interactive Interface: User-friendly forms to enter health parameters.

Model Insights: Displays prediction results along with model confidence scores.

Visualization: Graphical representation of feature importance and prediction probabilities.

🛠️ Tech Stack

Programming Language: Python

Machine Learning Libraries: Scikit-learn, XGBoost

Web Framework: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Serialization: joblib

Deployment: Streamlit Cloud

📁 Folder Structure
AI_Disease_Predictor/
├── app/
│   └── streamlit_app.py       # Main Streamlit application
├── models/
│   ├── diabetes.pkl           # Trained model for Diabetes prediction
│   ├── heart_disease.pkl      # Trained model for Heart Disease prediction
│   └── parkinsons.pkl         # Trained model for Parkinson's prediction
├── notebooks/
│   ├── diabetes_model.ipynb   # Jupyter notebook for Diabetes model training
│   ├── heart_model.ipynb      # Jupyter notebook for Heart Disease model training
│   └── parkinsons_model.ipynb # Jupyter notebook for Parkinson's model training
├── data/
│   ├── diabetes.csv           # Dataset for Diabetes
│   ├── heart_disease.csv      # Dataset for Heart Disease
│   └── parkinsons.csv         # Dataset for Parkinson's
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/Vaenvoice/AI_Disease_Predictor.git
cd AI_Disease_Predictor

2. Set Up a Virtual Environment

For Windows:

python -m venv venv
venv\Scripts\activate


For macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application Locally
streamlit run app/streamlit_app.py


Open your browser and navigate to http://localhost:8501 to interact with the application.

📊 Model Training

The machine learning models are trained using the datasets located in the data/ folder. Jupyter notebooks in the notebooks/ directory provide step-by-step guidance on training models for each disease:

Diabetes: notebooks/diabetes_model.ipynb

Heart Disease: notebooks/heart_model.ipynb

Parkinson's: notebooks/parkinsons_model.ipynb

🌐 Deployment

The application is deployed on Streamlit Cloud
. Once deployed, you can access the live application via the provided URL.

📄 License

This project is licensed under the MIT License - see the LICENSE
 file for details.

📬 Contact

For any inquiries or contributions, please contact Vaenvoice
.
