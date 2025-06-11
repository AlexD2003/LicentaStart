# train_all_models_mias.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ======= Config =======
DATA_DIR = "datasets/tensor_dataset_mias/"
LABEL_CSV = "datasets/modified_labels.csv"
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Dataset Class =======
class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, label_csv, transform=None, target_size=(224, 224)):
        df = pd.read_csv(label_csv)
        self.data = df
        self.tensor_folder = tensor_folder
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0] + ".pt"
        label = int(self.data.iloc[idx, 1])
        path = os.path.join(self.tensor_folder, file_name)
        tensor = torch.load(path)
        tensor = TF.resize(tensor, self.target_size)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

# ======= SimpleCNN Model =======
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

# ======= Transforms =======
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])
val_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# ======= Load and Split Dataset =======
full_dataset = MammogramDataset(DATA_DIR, LABEL_CSV)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_data = MammogramDataset(DATA_DIR, LABEL_CSV, transform=train_transform)
val_data = MammogramDataset(DATA_DIR, LABEL_CSV, transform=val_transform)
train_data.data = full_dataset.data.iloc[train_subset.indices].reset_index(drop=True)
val_data.data = full_dataset.data.iloc[val_subset.indices].reset_index(drop=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ======= Class Weights =======
labels = pd.read_csv(LABEL_CSV)["label"]
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ======= Model List =======
model_configs = [
    ("SimpleCNN", SimpleCNN(), (256, 256), 0.001, "models/simplecnn_mias.pth"),
    ("EfficientNetB1", models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT), (240, 240), 1e-5, "models/efficientnet_b1_mias.pth"),
    ("ResNet18", models.resnet18(weights=models.ResNet18_Weights.DEFAULT), (224, 224), 1e-5, "models/resnet18_mias.pth"),
    ("ResNet34", models.resnet34(weights=models.ResNet34_Weights.DEFAULT), (224, 224), 1e-5, "models/resnet34_mias.pth"),
    ("ResNet50", models.resnet50(weights=models.ResNet50_Weights.DEFAULT), (224, 224), 1e-5, "models/resnet50_mias.pth"),
]

# ======= Training Loop =======
for model_name, model, target_size, lr, save_path in model_configs:
    print(f"\n🔵 Training {model_name}...")

    # Adjust model for grayscale input and 2-class output
    if "EfficientNet" in model_name:
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 2)
        )
    elif "ResNet" in model_name:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 2)
        )
    # SimpleCNN is already correct

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, targets in tqdm(train_loader, desc=f"{model_name} - Epoch {epoch+1}/{EPOCHS}"):
            # Resize for each model
            inputs = TF.resize(inputs, target_size)

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
                inputs = TF.resize(inputs, target_size)
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"\n📊 {model_name} Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best {model_name} model saved to {save_path}")

print("\n✅ All models trained and saved.")
