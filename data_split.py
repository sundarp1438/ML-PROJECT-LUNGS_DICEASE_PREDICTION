import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split

def split_data(config_path):
    """Splits dataset into train and test sets."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_path = config.get('featurize', {}).get('processed_path', 'data/processed/featured_lung_disease_data.csv')
    train_path = config.get('data_split', {}).get('trainset_path', 'data/processed/train_lung_disease_data.csv')
    test_path = config.get('data_split', {}).get('testset_path', 'data/processed/test_lung_disease_data.csv')
    test_size = config.get('data_split', {}).get('test_size', 0.2)
    random_state = config.get('base', {}).get('random_state', 42)

    # Load dataset
    df = pd.read_csv(data_path)

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train set saved at {train_path}")
    print(f"✅ Test set saved at {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    split_data(args.config)