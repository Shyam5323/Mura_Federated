import os
import pandas as pd

# Directory where partitions are stored
PARTITIONS_DIR = "partitions"
STRATEGIES = ["iid", "pathological_non_iid", "label_skew"]

def expand_study_to_images(input_csv, output_csv):
    """
    Converts a study-level CSV to image-level CSV.
    Each row in the input CSV should have 'path' and 'label'.
    """
    df = pd.read_csv(input_csv)
    rows = []

    for _, row in df.iterrows():
        folder = row['path']
        label = row['label']
        if not os.path.exists(folder):
            print(f"Warning: Folder does not exist: {folder}")
            continue
        for file in os.listdir(folder):
            if file.startswith("._"):
                continue
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append([os.path.join(folder, file), label])

    new_df = pd.DataFrame(rows, columns=['path', 'label'])
    new_df.to_csv(output_csv, index=False)
    print(f"Saved {len(new_df)} images to {output_csv}")

def process_all_clients():
    for strategy in STRATEGIES:
        strategy_dir = os.path.join(PARTITIONS_DIR, strategy)
        if not os.path.exists(strategy_dir):
            print(f"Strategy folder not found: {strategy_dir}, skipping.")
            continue

        for file in os.listdir(strategy_dir):
            if file.endswith(".csv"):
                input_csv = os.path.join(strategy_dir, file)
                output_csv = os.path.join(strategy_dir, file.replace(".csv", "_images.csv"))
                expand_study_to_images(input_csv, output_csv)

if __name__ == "__main__":
    process_all_clients()
