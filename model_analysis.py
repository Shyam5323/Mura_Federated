import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score

# --- Configuration ---
MODEL_PATH = "best_resnet18_mura.pth"
MURA_FOLDER = 'MURA-v1.1'
VALID_CSV_PATH = os.path.join(MURA_FOLDER, 'valid_image_paths.csv')
VALID_INFO_CSV_PATH = os.path.join(MURA_FOLDER, 'valid_labeled_studies.csv') # CSV with body part info
BATCH_SIZE = 64 # Can be larger for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class (copied from your training script) ---
class MuraDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file, header=None, names=['path'])
        self.transform = transform
        self.labels = [1 if 'positive' in path else 0 for path in self.dataframe['path']]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        # Prepend the main folder path if it's not in the CSV
        if not img_path.startswith(MURA_FOLDER):
             img_path = os.path.join(MURA_FOLDER, img_path.split('MURA-v1.1/')[-1])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Main Analysis Functions ---

def get_all_preds_and_labels(model, dataloader):
    """Runs inference and returns probabilities and true labels."""
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            # Your model has 1 output, so use sigmoid
            probs = torch.sigmoid(outputs).squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

def find_best_threshold(probs, labels):
    """Finds the best threshold to maximize the F1-score."""
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.linspace(0, 1, 101):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def analyze_by_body_part(probs, labels, dataloader, best_threshold):
    """Computes and prints detailed metrics for each body part."""
    print("\n--- Analysis by Body Part ---")
    
    # Create a DataFrame with paths, labels, and predictions
    df = dataloader.dataset.dataframe.copy()
    df['label'] = labels
    df['probability'] = probs
    df['prediction'] = (probs >= best_threshold).astype(int)
    
    # Extract body part from the image path
    # Example path: MURA-v1.1/train/XR_SHOULDER/patient00001/...
    df['body_part'] = df['path'].apply(lambda x: x.split('/')[2].replace('XR_', ''))
    
    results = []
    for part in sorted(df['body_part'].unique()):
        part_df = df[df['body_part'] == part]
        y_true = part_df['label']
        y_pred = part_df['prediction']
        
        results.append({
            "Body Part": part,
            "F1": f1_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Support": len(y_true),
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_dataset = MuraDataset(csv_file=VALID_CSV_PATH, transform=valid_transforms)
    # IMPORTANT: shuffle=False to keep order for body part analysis
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model Architecture and Weights
    # The architecture MUST EXACTLY match the one you trained
    model = resnet18(weights=None) # No pre-trained weights needed here
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")

    # 3. Get Predictions
    all_probs, all_labels = get_all_preds_and_labels(model, valid_loader)

    # 4. Find Best Threshold
    best_threshold, best_f1 = find_best_threshold(all_probs, all_labels)
    print(f"\n✅ Best Threshold Found: {best_threshold:.2f} (achieves F1-Score: {best_f1:.4f})")
    
    # 5. Generate Final Report and Confusion Matrix
    final_preds = (all_probs >= best_threshold).astype(int)
    print("\n--- Classification Report (at optimal threshold) ---")
    print(classification_report(all_labels, final_preds, target_names=["Negative", "Positive"]))
    
    cm = confusion_matrix(all_labels, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix @ Threshold = {best_threshold:.2f}")
    plt.show()

    # 6. Perform Per-Body-Part Analysis
    analyze_by_body_part(all_probs, all_labels, valid_loader, best_threshold)