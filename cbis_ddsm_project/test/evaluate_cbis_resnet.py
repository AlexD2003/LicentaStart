import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b1
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

# === Dataset
class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, label_file, transform=None, target_size=(240, 240)):
        self.tensor_folder = tensor_folder
        self.transform = transform
        self.target_size = target_size

        df = pd.read_csv(label_file)
        df["image_id"] = df["image_id"].str.replace('"', '').str.replace(".png", "")
        def normalize(name): return re.sub(r'[^A-Z0-9]', '', name.upper())
        df["norm_id"] = df["image_id"].apply(normalize)

        pt_files = [f for f in os.listdir(tensor_folder) if f.endswith(".pt")]
        pt_map = {normalize(f.replace(".pt", "")): f for f in pt_files}

        matched = []
        for _, row in df.iterrows():
            norm_id = row["norm_id"]
            if norm_id in pt_map:
                matched.append({
                    "tensor_file": pt_map[norm_id],
                    "label": int(row["label"]),
                    "raw_id": row["image_id"]
                })

        self.labels_df = pd.DataFrame(matched)
        print(f"✅ Using {len(self.labels_df)} matched test samples.")

    def __len__(self): return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        path = os.path.join(self.tensor_folder, row["tensor_file"])
        tensor = torch.load(path)
        tensor = TF.resize(tensor, self.target_size)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(row["label"], dtype=torch.long), row["tensor_file"]

# === Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_folder = "datasets/tensor_dataset_test"
label_file = "csvs/test_labels.csv"
model_path = "models/cbis_efficientnet_b1.pth"
transform = transforms.Normalize((0.5,), (0.5,))

# === Load dataset
dataset = MammogramDataset(tensor_folder, label_file, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Load model
model = efficientnet_b1()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Inference
probs, labels, filenames = [], [], []

with torch.no_grad():
    for x, y, fname in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        p = torch.softmax(output, dim=1)[:, 1].item()
        probs.append(p)
        labels.append(y.item())
        filenames.append(fname[0])

# === Convert to arrays
probs = np.array(probs)
labels = np.array(labels)

# === Threshold Sweep
print("\n🎯 EfficientNet-B1 Threshold Sweep:")
print("Thresh | Prec   | Recall | F1     | Acc")
print("-------|--------|--------|--------|--------")

for t in np.linspace(0.1, 0.9, 17):
    preds = (probs >= t).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = (preds == labels).mean()
    print(f"{t:6.2f} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {acc:6.2f}")
# Per-sample predictions
print("\n📄 Individual Predictions:")
for fname, prob, label in zip(filenames, probs, labels):
    pred_label = "Malignant" if prob >= 0.5 else "Benign"
    true_label = "Malignant" if label == 1 else "Benign"
    print(f"{fname:<30} | Pred: {pred_label:<10} | Actual: {true_label:<10} | Prob(M): {prob:.2f}")
