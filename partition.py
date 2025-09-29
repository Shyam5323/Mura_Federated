import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_body_part_from_path(path_string: str) -> str:
    """
    Extracts the body part (e.g., 'XR_WRIST') from a MURA dataset path string.
    """
    parts = str(path_string).split('/')
    for part in parts:
        if 'XR_' in part:
            return part
    return "UNKNOWN"

def load_mura_dataframe(csv_path: str, base_dir: str) -> pd.DataFrame:
    """
    Loads a MURA dataset CSV into a pandas DataFrame and extracts body parts.
    """
    full_csv_path = os.path.join(base_dir, csv_path)
    if not os.path.exists(full_csv_path):
        print(f"Error: CSV file not found at '{full_csv_path}'")
        return None
    
    df = pd.read_csv(full_csv_path, names=['path', 'label'])
    df['body_part'] = df['path'].apply(get_body_part_from_path)
    return df

def report_and_save_partitions(partitions: dict, strategy_name: str):
    """
    Prints a summary of the data distribution for each client and saves partitions to CSVs.
    """
    print(f"\n--- Reporting for Partition Strategy: {strategy_name} ---")
    output_dir = os.path.join('partitions', strategy_name)
    os.makedirs(output_dir, exist_ok=True)

    for client_id, data in partitions.items():
        client_file = os.path.join(output_dir, f'client_{client_id}.csv')
        data.to_csv(client_file, index=False)
        
        print(f"\nClient {client_id}:")
        print(f"  Total studies: {len(data)}")
        if not data.empty:
            print(f"  Saved to: {client_file}")
            print("  Label distribution:")
            print(data['label'].value_counts().rename({0: 'Normal', 1: 'Abnormal'}).to_string())
            print("  Body part distribution:")
            print(data['body_part'].value_counts().to_string())
            
    print("\n" + "="*60 + "\n")

# --- Partitioning Strategies ---

def partition_iid(df: pd.DataFrame, num_clients: int) -> dict:
    """
    Strategy 1: IID Partitioning.
    Shuffles the dataset and deals it out evenly to each client.
    """
    print("\nPartitioning data using IID strategy...")
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    partitions = np.array_split(df_shuffled, num_clients)
    return {i: partitions[i] for i in range(num_clients)}

def partition_by_body_part(df: pd.DataFrame) -> dict:
    """
    Strategy 2: Pathological Non-IID Partitioning.
    Each client gets all data for one specific body part.
    """
    print("\nPartitioning data by body part (Pathological Non-IID)...")
    body_parts = df['body_part'].unique()
    return {part: df[df['body_part'] == part] for part in body_parts}

def partition_by_label_skew(df: pd.DataFrame, num_clients: int, alpha: float = 0.5) -> dict:
    """
    Strategy 3: Label Distribution Skew (Dirichlet).
    Each client gets a mix of body parts but a skewed ratio of normal/abnormal labels.
    'alpha' controls the skewness: lower alpha = more skew.
    """
    print(f"\nPartitioning data by label skew (Dirichlet with alpha={alpha})...")
    
    labels = df['label'].unique()
    partitions = {i: pd.DataFrame() for i in range(num_clients)}
    
    for label in labels:
        label_df = df[df['label'] == label]
        
        # Use Dirichlet distribution to get proportions for this label
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Distribute the data for this label among clients
        label_splits = np.split(
            label_df.sample(frac=1),
            (np.cumsum(proportions) * len(label_df)).astype(int)[:-1]
        )
        
        for i, split in enumerate(label_splits):
            partitions[i] = pd.concat([partitions[i], split])
            
    return partitions


if __name__ == '__main__':
    BASE_DATA_DIR = 'MURA-v1.1'
    NUM_CLIENTS = 7 

    print("Loading MURA training data...")
    train_df = load_mura_dataframe('train_labeled_studies.csv', BASE_DATA_DIR)

    if train_df is not None:
        # Strategy 1: IID
        iid_partitions = partition_iid(train_df, NUM_CLIENTS)
        report_and_save_partitions(iid_partitions, "iid")

        # Strategy 2: Pathological Non-IID (by body part)
        body_part_partitions = partition_by_body_part(train_df)
        report_and_save_partitions(body_part_partitions, "pathological_non_iid")

        # Strategy 3: Label Skew
        label_skew_partitions = partition_by_label_skew(train_df, NUM_CLIENTS, alpha=0.5)
        report_and_save_partitions(label_skew_partitions, "label_skew")
