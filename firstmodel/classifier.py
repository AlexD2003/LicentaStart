import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Custom dataset
class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, label_file, indices=None, transform=None, target_size=(256, 256)):
        self.tensor_folder = tensor_folder
        df = pd.read_csv(label_file, dtype={"image_id": str, "label": int})
        self.labels_df = df if indices is None else df.iloc[indices].reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        file_name = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])
        tensor_path = os.path.join(self.tensor_folder, file_name + ".pt")
        image_tensor = torch.load(tensor_path)  # Should already be scaled 0–1
        image_tensor = TF.resize(image_tensor, self.target_size)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, torch.tensor(label, dtype=torch.long)

# Improved CNN model
class MammogramCNN(nn.Module):
    def __init__(self):
        super(MammogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flattened_size = 64 * (256 // 4) * (256 // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Settings
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001
TENSOR_DIR = "datasets/tensor_dataset_mias/"
LABELS_FILE = "datasets/modified_labels.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stratified split
labels_df = pd.read_csv(LABELS_FILE)
indices = list(range(len(labels_df)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels_df["label"], random_state=42)

# Transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])
transform_val = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets and loaders
train_dataset = MammogramDataset(TENSOR_DIR, LABELS_FILE, indices=train_idx, transform=transform_train)
val_dataset = MammogramDataset(TENSOR_DIR, LABELS_FILE, indices=val_idx, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
model = MammogramCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "mammogram_cnn_mias.pth")
print("✅ Training complete and model saved.")
