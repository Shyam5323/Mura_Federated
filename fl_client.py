import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import numpy as np
from collections import OrderedDict
import flwr as fl
from sklearn.metrics import roc_auc_score
import argparse
import os


class MuraClientDataset(Dataset):
    """Dataset for federated client - loads from partition CSV"""
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        
        if 'path' not in df.columns:
            with open(csv_file, 'r') as f:
                paths = [line.strip() for line in f.readlines()]
                data = []
            for path in paths:
                label = 1 if 'positive' in path else 0
                data.append({'path': path, 'label': label})
            
            self.dataframe = pd.DataFrame(data)
        else:
            self.dataframe = df
            
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = int(row['label'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping {img_path} ({e})")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_weighted_sampler(labels):
    """Creates WeightedRandomSampler for class imbalance"""
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Evaluate model"""
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


class MuraFlowerClient(fl.client.NumPyClient):
    """Flower client for MURA federated learning"""
    
    def __init__(self, client_id, partition_path, valid_csv, device, batch_size=32, local_epochs=1):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Data transforms
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
        
        # Load datasets
        print(f"Client {client_id}: Loading data from {partition_path}")
        train_dataset = MuraClientDataset(partition_path, transform=train_transforms)
        valid_dataset = MuraClientDataset(valid_csv, transform=valid_transforms)
        
        # Create weighted sampler for training
        train_labels = [train_dataset.dataframe.iloc[i]['label'] for i in range(len(train_dataset))]
        train_sampler = get_weighted_sampler(train_labels)
        
        self.trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        self.valloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Model setup
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
        self.model = self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        print(f"Client {client_id}: Training samples = {len(train_dataset)}, "
              f"Validation samples = {len(valid_dataset)}")
    
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on local data"""
        self.set_parameters(parameters)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        print(f"\n[Client {self.client_id}] Starting local training...")
        
        for epoch in range(self.local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model, self.trainloader, self.criterion, optimizer, self.device
            )
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs}: "
                  f"Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on validation data"""
        self.set_parameters(parameters)
        
        val_loss, val_acc, val_auc = evaluate_model(
            self.model, self.valloader, self.criterion, self.device
        )
        
        print(f"[Client {self.client_id}] Validation: Loss={val_loss:.4f}, "
              f"Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
        
        return float(val_loss), len(self.valloader.dataset), {
            "accuracy": float(val_acc),
            "auc": float(val_auc)
        }


def main():
    parser = argparse.ArgumentParser(description='Flower Client for MURA')
    parser.add_argument('--client_id', type=str, required=True, help='Client identifier')
    parser.add_argument('--partition_strategy', type=str, required=True, 
                        choices=['iid', 'pathological_non_iid', 'label_skew'],
                        help='Partitioning strategy')
    parser.add_argument('--server_address', type=str, default='localhost:8080',
                        help='Server address')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--local_epochs', type=int, default=1, help='Local training epochs')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    partition_path = os.path.join('partitions', args.partition_strategy, f'client_{args.client_id}_images.csv')
    valid_csv = os.path.join('MURA-v1.1', 'valid_image_paths.csv')
    
    # Check if files exist
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"Partition file not found: {partition_path}")
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Validation file not found: {valid_csv}")
    
    # Create and start client
    client = MuraFlowerClient(
        client_id=args.client_id,
        partition_path=partition_path,
        valid_csv=valid_csv,
        device=device,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs
    )
    
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()