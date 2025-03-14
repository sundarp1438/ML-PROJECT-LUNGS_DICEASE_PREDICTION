import os
import json
import joblib
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(config_path):
    """Evaluates the trained model and saves metrics, including confusion matrix visualization."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    test_path = config.get('data_split', {}).get('testset_path', 'data/processed/test_lung_disease_data.csv')
    model_path = config.get('train', {}).get('model_path', 'models/model.joblib')
    metrics_path = config.get('evaluate', {}).get('metrics_path', 'reports/metrics.json')
    cm_path = config.get('evaluate', {}).get('cm_path', 'reports/confusion_matrix.png')
    target_column = config.get('featurize', {}).get('target_column', 'Recovered')

    # ✅ Check if test dataset exists
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ ERROR: Test dataset file '{test_path}' not found!")

    # ✅ Check if trained model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ ERROR: Trained model file '{model_path}' not found!")

    # ✅ Load test dataset
    df = pd.read_csv(test_path)

    # ✅ Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"❌ ERROR: Target column '{target_column}' is missing in the dataset!")

    # ✅ Ensure categorical features match training (if used)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"⚠️ WARNING: Found categorical columns {categorical_cols}. Converting to numeric...")
        df = pd.get_dummies(df, drop_first=True)  # One-hot encoding

    # ✅ Split features & target
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # ✅ Load trained model
    model = joblib.load(model_path)

    # ✅ Make predictions
    y_pred = model.predict(X_test)

    # ✅ Compute metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ✅ Ensure output directories exist
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)

    # ✅ Save metrics as JSON
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # ✅ Plot confusion matrix with improved labels
    plt.figure(figsize=(6, 4))
    labels = sorted(set(y_test))  # Ensure correct order of labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # ✅ Save confusion matrix as PNG
    plt.savefig(cm_path)
    plt.close()

    print(f"✅ Evaluation complete. Metrics saved at: {metrics_path}")
    print(f"✅ Confusion matrix saved at: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    evaluate_model(args.config)
