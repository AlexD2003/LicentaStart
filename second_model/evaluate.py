import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import MammogramDataset
from model.model import ResNet18Classifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Config
MODEL_PATH = "resnet18_mammogram.pth"
DATA_DIR = "datasets/tensor_dataset_mias_test/"
LABEL_CSV = "datasets/test_labels.csv"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
dataset = MammogramDataset(DATA_DIR, LABEL_CSV, transform=transform, target_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = ResNet18Classifier(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Inference
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
