import pandas as pd
import os

def load_diabetes_data():
    """
    Loads the Pima Indians Diabetes dataset.
    """

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
    ]

    df = pd.read_csv(url, header=None, names=columns)

    # Save raw copy
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/diabetes.csv", index=False)

    return df


if __name__ == "__main__":
    df = load_diabetes_data()
    print("Dataset shape:", df.shape)
    print(df.head())
