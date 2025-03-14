import os
import joblib
import yaml
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently import ColumnMapping
import warnings

warnings.filterwarnings("ignore")

# ✅ Load Configuration
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

# ✅ Load MLflow model OR local model
use_mlflow = True  # Change to False to use a local model

if use_mlflow:
    logged_model = 'runs:/79b10b825f1f48609c26a4b22dac59e7/model'  # Use the latest MLflow model
    model = mlflow.pyfunc.load_model(logged_model)
    print("✅ Model loaded from MLflow")
else:
    model_path = config.get("train", {}).get("model_path", "models/lung_disease_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")

# ✅ Load feature names from training dataset
train_data_path = config["data_split"]["trainset_path"]

if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"🚨 Training data not found at {train_data_path}")

train_df = pd.read_csv(train_data_path)

# ✅ Remove the target column to get feature names
target = config["featurize"]["target_column"]
feature_columns = [col for col in train_df.columns if col != target]

# ✅ Load Lung Disease Data (Reference: Train, Current: Test)
data_paths = {
    "reference": config["data_split"]["trainset_path"],  # Train Data
    "current": config["data_split"]["testset_path"]  # Test Data
}

datasets = {}
for key, path in data_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"🚨 Data file not found: {path}")
    datasets[key] = pd.read_csv(path)
    print(f"✅ {key.capitalize()} data loaded from {path}")

# ✅ Check for missing columns before prediction
for key, df in datasets.items():
    print(f"\n🔍 Checking columns in {key} dataset:")
    print(df.columns.tolist())  # Debugging: print available columns

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"🚨 Missing columns in '{key}' dataset: {missing_columns}")

    # ✅ Ensure column types are numeric
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

    # ✅ Make Predictions
    datasets[key]["prediction"] = model.predict(df[feature_columns])

# ✅ Define Column Mapping for Evidently AI
column_mapping = ColumnMapping()
column_mapping.target = target
column_mapping.prediction = "prediction"
column_mapping.numerical_features = feature_columns
column_mapping.categorical_features = ["Gender", "Smoking Status", "Disease Type", "Treatment Type"]

# ✅ Run Evidently Data Drift Report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset()
])
data_drift_report.run(reference_data=datasets["reference"], current_data=datasets["current"], column_mapping=column_mapping)

# ✅ Save Report
drift_report_path = "reports/lung_disease_drift_report.html"
os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
data_drift_report.save_html(drift_report_path)
print(f"✅ Data drift report saved at {drift_report_path}")

# ✅ Log Report to MLflow
mlflow.set_experiment("Lung Disease Data Drift Monitoring")

with mlflow.start_run():
    mlflow.log_artifact(drift_report_path)
    print("✅ Drift report logged to MLflow")

print("\n🎯 Lung Disease Data Drift Analysis Complete. Check MLflow UI for logs.")
