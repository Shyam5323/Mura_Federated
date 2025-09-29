import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import copy


def freeze_all_but_fc(model):
    """Freezes all layers of the model except for the final fully connected layer."""
    print("🧊 Freezing all layers except the final classifier head.")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_last_block(model):
    """Unfreezes the final convolutional block (layer4) and the classifier head."""
    print("🔓 Unfreezing last block (layer4 + FC).")
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    """Unfreezes all layers of the model."""
    print("🔓 Unfreezing all layers for full model fine-tuning.")
    for param in model.parameters():
        param.requires_grad = True

def get_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler to handle class imbalance by oversampling
    the minority class, ensuring each batch has a balanced distribution.
    """
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# --- 2. Custom Dataset Definition ---
class MuraDataset(Dataset):
    """
    Custom PyTorch Dataset for the MURA dataset.
    It loads images from paths specified in a CSV file.
    """
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file, header=None, names=['path'])
        self.transform = transform
        self.labels = [1 if 'positive' in path else 0 for path in self.dataframe['path']]
        print("Val positives:", sum(self.labels), "Val negatives:", len(self.labels) - sum(self.labels))
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 3. Model Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on the validation set and computes AUC."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels_float = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels_float)
            running_loss += loss.item() * inputs.size(0)
            preds_sigmoid = torch.sigmoid(outputs)
            preds_binary = preds_sigmoid > 0.5
            correct_predictions += (preds_binary == labels_float).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds_sigmoid.cpu().numpy().flatten())

    val_loss = running_loss / total_samples
    val_acc = (correct_predictions / total_samples) * 100
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    return val_loss, val_acc, auc_score

# --- 4. Main Execution Block ---

if __name__ == '__main__':
    # --- Configuration ---
    QUICK_TEST = False 
    QUICK_TEST_FRAC = 0.2
    MURA_FOLDER = 'MURA-v1.1'
    # Increased epochs to accommodate the new schedule
    NUM_EPOCHS = 6
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Preparation with AUGMENTATION ---
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- PHASE 1: CENTRALIZED BENCHMARK ---
    print("\n--- Starting Phase 1: Centralized Benchmark Training ---")
    
    full_train_dataset = MuraDataset(
        csv_file=os.path.join(MURA_FOLDER, 'train_image_paths.csv'),
        transform=train_transforms
    )
    valid_dataset = MuraDataset(
        csv_file=os.path.join(MURA_FOLDER, 'valid_image_paths.csv'),
        transform=valid_transforms
    )
    
    train_subset_for_sampler = full_train_dataset
    if QUICK_TEST:
        print(f"--- QUICK TEST MODE: Using {QUICK_TEST_FRAC*100}% of the data ---")
        num_train_samples = int(len(full_train_dataset) * QUICK_TEST_FRAC)
        full_train_dataset = Subset(full_train_dataset, range(num_train_samples))
        train_subset_for_sampler = full_train_dataset
        num_valid_samples = int(len(valid_dataset) * QUICK_TEST_FRAC)
        valid_dataset = Subset(valid_dataset, range(num_valid_samples))

    # --- Handling Class Imbalance with WeightedRandomSampler ---
    print("Setting up WeightedRandomSampler to handle class imbalance...")
    train_labels = np.array(train_subset_for_sampler.dataset.labels)[train_subset_for_sampler.indices] if QUICK_TEST else train_subset_for_sampler.labels
    train_sampler = get_weighted_sampler(train_labels)
    
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Setup with Dropout ---
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop with Gradual Unfreezing ---
    best_val_auc = 0.0
    best_model_wts = None
    
    # Initial phase: Train only the classifier head
    freeze_all_but_fc(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(NUM_EPOCHS):
        # Granular Unfreezing Schedule
        if epoch == 2:
            unfreeze_last_block(model)
            # Re-initialize optimizer for the newly trainable parameters with a smaller LR
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        if epoch == 4:
            unfreeze_all(model)
            # Re-initialize optimizer for the full model with a very small LR
            optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auc = evaluate_model(model, valid_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved with Val AUC: {best_val_auc:.4f}")

    print("\nCentralized training complete.")
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model with Final Validation AUC: {best_val_auc:.4f}")
        # --- ADD THESE TWO LINES ---
        torch.save(best_model_wts, "best_resnet18_mura.pth")
        print("✅ Best model weights saved to best_resnet18_mura.pth")

