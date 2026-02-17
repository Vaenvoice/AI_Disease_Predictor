import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_disease_model(disease_name):
    disease_name = disease_name.lower()
    
    # 1. Load the correct processed data
    try:
        X_train = pd.read_csv(f"data/processed/{disease_name}_X_train.csv")
        X_test = pd.read_csv(f"data/processed/{disease_name}_X_test.csv")
        y_train = pd.read_csv(f"data/processed/{disease_name}_y_train.csv").values.ravel()
        y_test = pd.read_csv(f"data/processed/{disease_name}_y_test.csv").values.ravel()
    except FileNotFoundError:
        print(f"Error: Processed data for {disease_name} not found. Run preprocess.py first.")
        return

    # 2. Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True)
    }

    results = {}

    # 3. Train & Evaluate
    print(f"\nTraining models for {disease_name.upper()}...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        }

    results_df = pd.DataFrame(results).T
    
    # 4. Select and Save Best Model
    best_model_name = results_df["ROC AUC"].idxmax()
    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{disease_name}_best_model.pkl")

    print(results_df)
    print(f"Best Model for {disease_name}: {best_model_name} (Saved!)")

if __name__ == "__main__":
    disease = input("Enter disease to train (diabetes/heart/kidney/parkinsons/all): ").lower()
    
    if disease == "all":
        for d in ["diabetes", "heart", "kidney", "parkinsons"]:
            train_disease_model(d)
    else:
        train_disease_model(disease)
