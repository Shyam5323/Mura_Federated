import pandas as pd
import os

def get_body_part_from_path(path_string: str) -> str:
    parts = str(path_string).split('/')
    for part in parts:
        if 'XR_' in part:
            return part
    return "UNKNOWN" 

def analyze_mura_from_csv(csv_path: str, base_dir: str):
    full_csv_path = os.path.join(base_dir, csv_path)

    if not os.path.exists(full_csv_path):
        print(f"Error: CSV file not found at '{full_csv_path}'")
        print("Please make sure the 'base_data_dir' variable points to your 'MURA-v1.1' folder.")
        return

    print(f"--- Analyzing Dataset from: {csv_path} ---")

    df = pd.read_csv(full_csv_path, names=['path', 'label'])

    df['body_part'] = df['path'].apply(get_body_part_from_path)

    stats = df.groupby('body_part')['label'].value_counts().unstack(fill_value=0)
    stats.columns = ['normal', 'abnormal'] 

    stats['total_studies'] = stats['normal'] + stats['abnormal']
    stats['abnormal_percentage'] = (stats['abnormal'] / stats['total_studies'] * 100).round(2)

    total_row = stats.sum(numeric_only=True)
    total_row['abnormal_percentage'] = (total_row['abnormal'] / total_row['total_studies'] * 100).round(2)
    total_row.name = 'TOTAL'
    stats = pd.concat([stats, pd.DataFrame(total_row).T])

    print(stats)
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    base_data_dir = 'MURA-v1.1' 

    analyze_mura_from_csv('train_labeled_studies.csv', base_data_dir)

    analyze_mura_from_csv('valid_labeled_studies.csv', base_data_dir)

