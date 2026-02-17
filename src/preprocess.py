import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================
# HELPERS
# ==============================

def parse_range(value):
    """Safely converts range strings like '1.019 - 1.021' to its mean."""
    if pd.isna(value) or value == "?":
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if '-' in value:
            try:
                parts = [float(i.strip()) for i in value.split('-')]
                return sum(parts) / len(parts)
            except (ValueError, IndexError):
                return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return value  # Keep as string for LabelEncoder

def preprocess(disease_name):
    disease_name = disease_name.lower().strip()
    if disease_name == "diabetes": return preprocess_diabetes()
    elif disease_name == "heart": return preprocess_heart()
    elif disease_name == "kidney": return preprocess_kidney()
    elif disease_name == "parkinsons": return preprocess_parkinsons()
    else: raise ValueError(f"Unsupported disease: {disease_name}")

# ==============================
# KIDNEY (THE TROUBLEMAKER)
# ==============================

def preprocess_kidney():
    df = pd.read_csv("data/raw/kidney.csv")
    df.columns = df.columns.str.strip()

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # 1. Clean ranges and '?'
    # Use .map() for Pandas 2.1+ / Python 3.14
    df = df.map(parse_range)

    # 2. Clean Target Column
    target_col = "class"
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        mapping = {'ckd': 1, 'notckd': 0, 'ckd\t': 1, 'notckd\t': 0}
        df[target_col] = df[target_col].replace(mapping)
        # Drop any row that isn't 0 or 1
        df = df[df[target_col].isin(['0', '1', 0, 1])]
        df[target_col] = pd.to_numeric(df[target_col]).astype(int)

    # 3. Clean Features
    for col in df.columns:
        if col == target_col: continue
        
        # Try to convert to numeric, if it fails, it's categorical
        converted = pd.to_numeric(df[col], errors='coerce')
        
        if not converted.isna().all():
            # It's numeric (at least partially)
            df[col] = converted.fillna(converted.median())
        else:
            # It's categorical
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return finalize_dataset(df, target_col, "kidney")

# ==============================
# OTHER DISEASES
# ==============================

def preprocess_diabetes():
    df = pd.read_csv("data/raw/diabetes.csv")
    df.columns = df.columns.str.strip()
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    return finalize_dataset(df, "Outcome", "diabetes")

def preprocess_heart():
    df = pd.read_csv("data/raw/heart.csv")
    df.columns = df.columns.str.strip()
    return finalize_dataset(df, "target", "heart")

def preprocess_parkinsons():
    df = pd.read_csv("data/raw/parkinsons.csv")
    df.columns = df.columns.str.strip()
    if "name" in df.columns: df.drop("name", axis=1, inplace=True)
    return finalize_dataset(df, "status", "parkinsons")

# ==============================
# FINAL STEP
# ==============================

def finalize_dataset(df, target_column, disease_name):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Use 'coerce' to turn everything to NaN, then fill with 0
    # This ensures StandardScaler ONLY gets floats
    X_final = X.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"data/processed/{disease_name}_X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"data/processed/{disease_name}_X_test.csv", index=False)
    y_train.to_csv(f"data/processed/{disease_name}_y_train.csv", index=False)
    y_test.to_csv(f"data/processed/{disease_name}_y_test.csv", index=False)
    joblib.dump(scaler, f"models/{disease_name}_scaler.pkl")

    print(f"\nSUCCESS: {disease_name.upper()} processed.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    name = input("Enter disease (diabetes/heart/kidney/parkinsons): ").lower().strip()
    try:
        preprocess(name)
    except Exception as e:
        print(f"Error: {e}")


