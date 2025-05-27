import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Config
LABELS_CSV = "csvs/train_labels.csv"
TENSOR_DIR = "datasets/tensor_dataset"
BATCH_SIZE = 8
EPOCHS = 20
MODEL_PATH = "models/cbis_model.pth"
DEVICE = torch.device("cpu")

# Dataset class
class MammogramDataset(Dataset):
    def __init__(self, csv_path, tensor_dir):
        self.data = pd.read_csv(csv_path)
        self.tensor_dir = tensor_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Clean up filename: strip quotes and convert .png to .pt
        fname = str(self.data.iloc[idx, 0]).replace('"', '').replace(".png", ".pt")
        label = int(self.data.iloc[idx, 1])
        path = os.path.join(self.tensor_dir, fname)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tensor: {path}")

        tensor = torch.load(path)  # shape: [1, 1024, 1024]
        tensor = TF.resize(tensor, [256, 256])  # Resize for memory efficiency
        return tensor, torch.tensor(label, dtype=torch.long)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Load dataset and split
full_dataset = MammogramDataset(LABELS_CSV, TENSOR_DIR)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Compute class weights
labels = pd.read_csv(LABELS_CSV)["label"]
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Model setup
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_acc = 100 * correct / total

    # Validation
    val_loss = 0
    val_correct, val_total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"\n📊 Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved!")

print(f"\n✅ Training complete. Best model saved to {MODEL_PATH}")
