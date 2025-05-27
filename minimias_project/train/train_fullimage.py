import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
from pathlib import Path

# === CONFIG ===
DATA_DIR = "datasets/tensor_fullimage"
MODEL_PATH = "models/fullimage_malignant.pth"
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 7
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
class FullImageDataset(Dataset):
    def __init__(self, split):
        self.path = Path(DATA_DIR) / split
        self.files = list(self.path.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        return sample["tensor"], torch.tensor(sample["label"], dtype=torch.long)

# === Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# === Transforms
transform = transforms.Normalize((0.5,), (0.5,))

# === Load Data
train_ds = FullImageDataset("train")
test_ds = FullImageDataset("test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Compute class weights
all_labels = [torch.load(f)["label"] for f in Path(DATA_DIR, "train").glob("*.pt")]
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=all_labels)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# === Model
model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model = model.to(DEVICE)

# === Training Setup
criterion = FocalLoss(gamma=2.0, weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
patience_counter = 0

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    total, correct, train_loss = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = transform(x)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    # === Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = transform(x)
            out = model(x)
            val_loss += criterion(out, y).item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

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
