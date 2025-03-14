import os
import pandas as pd
import yaml
import argparse
from sklearn.preprocessing import LabelEncoder

def featurize_data(config_path):
    """Applies feature engineering and saves processed dataset."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    raw_data_path = config.get('data_load', {}).get('dataset_path', 'data/raw/lung_disease_data.csv')
    processed_data_path = config.get('featurize', {}).get('processed_path', 'data/processed/featured_lung_disease_data.csv')

    # ✅ Ensure the raw dataset file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ ERROR: The dataset file '{raw_data_path}' does not exist!")

    # ✅ Load data
    df = pd.read_csv(raw_data_path)

    # ✅ Dynamically detect categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ✅ Apply Label Encoding to categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes  # More robust label encoding

    # ✅ Ensure the output directory exists before saving
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # ✅ Save processed data
    df.to_csv(processed_data_path, index=False)
    print(f"✅ Processed data saved at {processed_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    featurize_data(args.config)
