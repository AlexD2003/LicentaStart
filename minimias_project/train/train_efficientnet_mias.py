import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from tqdm import tqdm
import torchvision.transforms.functional as TF
from sklearn.utils.class_weight import compute_class_weight

# === CONFIG ===
CSV_PATH = "csvs/mias_train.csv"
TENSOR_DIR = "datasets/tensor_dataset_roi/train"
MODEL_PATH = "models/efficientnet_mias_focal.pth"
LABEL_CSV = "../csvs/mias_test.csv"
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device("cpu")
PATIENCE = 7
TARGET_SIZE = (240, 240)

# === Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# === Dataset
class ROIMiasDataset(Dataset):
    def __init__(self, csv_path, tensor_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["x", "y", "radius"])
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.samples = []
        id_counts = {}
        for _, row in self.df.iterrows():
            image_id = row["id"]
            count = id_counts.get(image_id, 0)
            filename = f"{image_id}_{count}.pt"
            label = 1 if row["label"] == "M" else 0
            self.samples.append((filename, label))
            id_counts[image_id] = count + 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.tensor_dir, fname)
        tensor = torch.load(path)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

# === Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Normalize((0.5,), (0.5,))

# === Load datasets
full_dataset = ROIMiasDataset(CSV_PATH, TENSOR_DIR)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(ROIMiasDataset(CSV_PATH, TENSOR_DIR, transform=train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ROIMiasDataset(CSV_PATH, TENSOR_DIR, transform=val_transform),
                        batch_size=BATCH_SIZE)

# === Class weights
labels = [1 if l == "M" else 0 for l in full_dataset.df["label"]]
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# === Model
model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
# ✅ Unfreeze all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# === Optimizer and Focal Loss
criterion = FocalLoss(gamma=2.0, weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# === Training loop
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total, correct, train_loss = 0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_loss += criterion(out, y).item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"\n📊 Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

print(f"\n✅ Training complete. Best model saved to {MODEL_PATH}")
