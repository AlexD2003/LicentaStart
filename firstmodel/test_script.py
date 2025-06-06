import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
import torchvision.transforms.functional as TF
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dataset
class MammogramDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_folder, label_file, transform=None, target_size=(256, 256)):
        self.tensor_folder = tensor_folder
        self.labels_df = pd.read_csv(label_file)
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        file_name = self.labels_df.iloc[idx, 0] + ".pt"
        label = int(self.labels_df.iloc[idx, 1])
        tensor_path = os.path.join(self.tensor_folder, file_name)
        image_tensor = torch.load(tensor_path)
        image_tensor = TF.resize(image_tensor, self.target_size)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, torch.tensor(label, dtype=torch.long)

# Model
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

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
test_dataset = MammogramDataset("datasets/tensor_dataset_mias_test/", "datasets/test_labels.csv", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
model = MammogramCNN().to(device)
model.load_state_dict(torch.load("mammogram_cnn_mias.pth"))
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Custom logging
malignant_total = sum(1 for l in all_labels if l == 1)
malignant_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l == 1)
print(f"\n🧬 Malignant cases in test set: {malignant_total}")
print(f"✅ Correctly detected malignant: {malignant_correct}")

# Metrics
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
