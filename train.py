import os
import pandas as pd
import yaml
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(config_path):
    """Trains a model and saves it."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_path = config.get('data_split', {}).get('trainset_path', 'data/processed/train_lung_disease_data.csv')
    model_path = config.get('train', {}).get('model_path', 'models/model.joblib')
    target_column = config.get('featurize', {}).get('target_column', 'Recovered')

    # ✅ Check if the train dataset exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ ERROR: The training dataset file '{train_path}' was not found!")

    # ✅ Load train dataset
    df = pd.read_csv(train_path)

    # ✅ Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"❌ ERROR: Target column '{target_column}' is missing in the dataset!")

    # ✅ Ensure all features are numeric
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"⚠️ WARNING: Found categorical columns {categorical_cols}. Converting to numeric...")
        X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical columns

    print(f"✅ Training dataset: {X.shape[0]} rows, {X.shape[1]} features.")

    # ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # ✅ Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ✅ Save model
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    train_model(args.config)
