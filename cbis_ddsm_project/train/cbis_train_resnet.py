import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models, transforms
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Config
LABELS_CSV = "csvs/train_labels.csv"
TENSOR_DIR = "datasets/tensor_dataset"
BATCH_SIZE = 8
EPOCHS = 20
MODEL_PATH = "models/cbis_resnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class MammogramDataset(Dataset):
    def __init__(self, csv_path, tensor_dir, transform=None):
        df = pd.read_csv(csv_path)
        df["image_id"] = df["image_id"].str.strip('"')
        self.data = df
        self.tensor_dir = tensor_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data.iloc[idx, 0].replace(".png", ".pt")
        label = int(self.data.iloc[idx, 1])
        path = os.path.join(self.tensor_dir, fname)

        tensor = torch.load(path)  # [1, 256, 256]
        tensor = TF.resize(tensor, [224, 224])
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# Load full dataset and split
full_dataset = MammogramDataset(LABELS_CSV, TENSOR_DIR)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# Wrap subsets with transform-aware datasets
train_set = [(TF.resize(torch.load(os.path.join(TENSOR_DIR, full_dataset.data.iloc[idx, 0].replace(".png", ".pt"))), [224, 224]), full_dataset.data.iloc[idx, 1]) for idx in train_subset.indices]
val_set = [(TF.resize(torch.load(os.path.join(TENSOR_DIR, full_dataset.data.iloc[idx, 0].replace(".png", ".pt"))), [224, 224]), full_dataset.data.iloc[idx, 1]) for idx in val_subset.indices]

# Create wrapped dataset class to apply transforms to subsets
class WrappedTensorDataset(Dataset):
    def __init__(self, base_data, transform=None):
        self.base_data = base_data
        self.transform = transform

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, idx):
        tensor, label = self.base_data[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(WrappedTensorDataset(train_set, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(WrappedTensorDataset(val_set, transform=val_transform), batch_size=BATCH_SIZE)

# Compute class weights
labels = pd.read_csv(LABELS_CSV)["label"]
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Load pretrained ResNet18 and adapt for grayscale
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"\n📊 Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved!")

print(f"\n✅ Training complete. Best model saved to {MODEL_PATH}")
