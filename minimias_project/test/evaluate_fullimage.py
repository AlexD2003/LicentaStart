import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b1
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# === CONFIG ===
DATA_DIR = "datasets/tensor_fullimage/test"
MODEL_PATH = "models/fullimage_malignant.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_INPUT_SIZE = 240

# === Dataset
class FullImageDataset(Dataset):
    def __init__(self, path):
        self.files = list(Path(path).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        return sample["tensor"], sample["label"]

# === Load model
model = efficientnet_b1()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Transforms
normalize = transforms.Normalize((0.5,), (0.5,))
tta_transforms = [
    transforms.Compose([normalize]),  # original
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), normalize]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), normalize]),
    transforms.Compose([transforms.RandomRotation(10), normalize]),
]

# === TTA Inference
loader = DataLoader(FullImageDataset(DATA_DIR), batch_size=1, shuffle=False)
probs, labels = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        pred_sum = 0
        for tf in tta_transforms:
            x_aug = tf(x.clone())
            out = model(x_aug)
            pred_sum += torch.softmax(out, dim=1)[:, 1]  # malignant prob
        avg_pred = (pred_sum / len(tta_transforms)).item()
        probs.append(avg_pred)
        labels.append(y.item())

probs = np.array(probs)
labels = np.array(labels)

# === Threshold Sweep
print("\n🎯 EfficientNet-B1 TTA Threshold Sweep:")
print("Thresh | Prec   | Recall | F1     | Acc")
print("-------|--------|--------|--------|--------")

for t in np.linspace(0.1, 0.9, 17):
    preds = (probs >= t).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = (preds == labels).mean()
    print(f"{t:6.2f} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {acc:6.2f}")
