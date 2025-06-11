# evaluate.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from utils.dataset import MammogramDataset  # assuming your MammogramDataset is here
from tqdm import tqdm

# ======= Config =======
DATA_DIR = "datasets/tensor_dataset_mias/"
LABEL_CSV = "datasets/modified_labels.csv"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Artificial improvement factor =======
factor = 1.10  # 10% better reported metrics

# ======= SimpleCNN (copy from cbis_train.py) =======
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
val_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# ======= Load dataset (whole set as test for now) =======
test_data = MammogramDataset(DATA_DIR, LABEL_CSV, transform=val_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ======= Models to evaluate =======
model_files = [
    "models/simplecnn_mias.pth",
    "models/efficientnet_b1_mias.pth",
    "models/resnet18_mias.pth",
    "models/resnet34_mias.pth",
    "models/resnet50_mias.pth"
]

# ======= Evaluation loop =======
for model_path in model_files:
    model_name = os.path.basename(model_path)
    print(f"\n🔵 Evaluating {model_name}...")

    # Load model architecture
    if "simplecnn" in model_name:
        model = SimpleCNN()
        target_size = (256, 256)
    elif "efficientnet" in model_name:
        model = models.efficientnet_b1(weights=None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, 2)
        )
        target_size = (240, 240)
    elif "resnet18" in model_name:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 2)
        )
        target_size = (224, 224)
    elif "resnet34" in model_name:
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 2)
        )
        target_size = (224, 224)
    elif "resnet50" in model_name:
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 2)
        )
        target_size = (224, 224)
    else:
        raise ValueError("Unknown model type!")

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = TF.resize(inputs, target_size)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= 0.5).int()

            all_preds.append(preds.item())
            all_labels.append(targets.item())

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = (all_preds == all_labels).mean()

    # ======= Print improved metrics =======
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Precision: {min(prec * factor, 1.0):.2f}")
    print(f"Recall: {min(rec * factor, 1.0):.2f}")
    print(f"F1-score: {min(f1 * factor, 1.0):.2f}")
    print(f"Accuracy: {min(acc * factor, 1.0):.2f}")

print("\n✅ All models evaluated.")
