import os
import pandas as pd
import yaml
import argparse
import requests

def load_data(config_path):
    """Loads dataset from a raw source and saves it as CSV."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load dataset path from config (defaulting to 'data/raw/lung_disease_data.csv')
    data_path = config.get('data_load', {}).get('dataset_path', 'data/raw/lung_disease_data.csv')

    # ✅ Correct GitHub raw file URL
    raw_url = "https://raw.githubusercontent.com/sundarp1438/ML-PROJECT-LUNGS_DICEASE_PREDICTION/main/lung_disease_data.csv"

    # ✅ Ensure the 'data/raw' directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    try:
        # ✅ Download the dataset using `requests`
        response = requests.get(raw_url)
        response.raise_for_status()  # Raise an error if the request failed

        # ✅ Save the dataset
        with open(data_path, "wb") as file:
            file.write(response.content)

        print(f"✅ Data successfully downloaded and saved at: {data_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")

    # ✅ Use `parse_known_args()` to avoid conflicts in Jupyter/Colab
    args, unknown = parser.parse_known_args()

    load_data(args.config)
