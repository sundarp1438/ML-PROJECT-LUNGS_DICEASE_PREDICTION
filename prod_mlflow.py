import os
import yaml
import argparse
import joblib
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_data(config):
    """Loads raw dataset from source and saves it."""
    raw_data_path = config["data_load"]["dataset_path"]
    processed_data_path = config["featurize"]["processed_path"]

    # ✅ Download dataset (if needed) - Modify URL if needed
    raw_url = "https://raw.githubusercontent.com/sundarp1438/ML-PROJECT-LUNGS_DICEASE_PREDICTION/main/lung_disease_data.csv"
    df = pd.read_csv(raw_url)

    # ✅ Ensure directories exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    df.to_csv(raw_data_path, index=False)

    mlflow.log_artifact(raw_data_path)

    print(f"✅ Data loaded and saved at {raw_data_path}")
    return df


def featurize_data(config, df):
    """Processes dataset by encoding categorical features."""
    processed_data_path = config["featurize"]["processed_path"]

    # ✅ Identify categorical columns & encode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # ✅ Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)

    mlflow.log_artifact(processed_data_path)

    print(f"✅ Processed data saved at {processed_data_path}")
    return df


def split_data(config, df):
    """Splits dataset into training & test sets."""
    train_path = config["data_split"]["trainset_path"]
    test_path = config["data_split"]["testset_path"]
    target_column = config["featurize"]["target_column"]
    test_size = config["data_split"]["test_size"]

    # ✅ Split dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config["base"]["random_state"])

    # ✅ Save split datasets
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)

    print(f"✅ Data split: Train set ({len(X_train)} rows), Test set ({len(X_test)} rows)")
    return X_train, X_test, y_train, y_test


def train_model(config, X_train, y_train):
    """Trains the model and logs to MLflow."""
    model_path = config["train"]["model_path"]

    # ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained and saved at {model_path}")
    return model


def evaluate_model(config, model, X_test, y_test):
    """Evaluates the model and logs metrics."""
    metrics_path = config["evaluate"]["metrics_path"]
    cm_path = config["evaluate"].get("cm_path", "reports/confusion_matrix.png")

    # ✅ Make predictions
    y_pred = model.predict(X_test)

    # ✅ Compute metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ✅ Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({"accuracy": acc, "confusion_matrix": cm.tolist()}, f, indent=4)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact(metrics_path)

    # ✅ Plot confusion matrix
    plt.figure(figsize=(6, 4))
    labels = sorted(set(y_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print(f"✅ Model evaluation complete. Accuracy: {acc}")
    return acc


def main(config_path):
    """Runs the entire MLflow pipeline."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Lung Disease Prediction Pipeline")

    with mlflow.start_run():
        df = load_data(config)
        df = featurize_data(config, df)
        X_train, X_test, y_train, y_test = split_data(config, df)
        model = train_model(config, X_train, y_train)
        evaluate_model(config, model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    main(args.config)
