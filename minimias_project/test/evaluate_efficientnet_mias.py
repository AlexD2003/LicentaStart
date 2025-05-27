import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b1
import torchvision.transforms.functional as TF
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

# === CONFIG ===
CSV_PATH = "csvs/mias_test.csv"
TENSOR_DIR = "datasets/tensor_dataset_roi/test"
MODEL_PATH = "models/efficientnet_mias.pth"
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
TARGET_SIZE = (240, 240)

# === Dataset ===
class MiasROITestDataset(Dataset):
    def __init__(self, csv_path, tensor_dir, transform=None):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["x", "y", "radius"])
        self.samples = []
        self.tensor_dir = tensor_dir
        self.transform = transform

        id_counts = {}
        for _, row in df.iterrows():
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

# === Load dataset
transform = transforms.Normalize((0.5,), (0.5,))
dataset = MiasROITestDataset(CSV_PATH, TENSOR_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load model
model = efficientnet_b1()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === Run inference
probs, labels = [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        p = torch.softmax(output, dim=1)[:, 1]  # malignant probability
        probs.extend(p.cpu().numpy())
        labels.extend(y.cpu().numpy())

probs = np.array(probs)
labels = np.array(labels)

# === Threshold sweep
print("\n🎯 Threshold Sweep:")
print("Thresh | Prec   | Recall | F1     | Acc")
print("-------|--------|--------|--------|--------")

for t in np.linspace(0.1, 0.9, 17):
    preds = (probs >= t).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = (preds == labels).mean()
    print(f"{t:6.2f} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {acc:6.2f}")
