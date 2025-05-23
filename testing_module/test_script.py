import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import pandas as pd
import os
import torchvision.transforms.functional as TF
from torch import nn
import random

# Dataset class
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
        image_tensor = torch.load(tensor_path)  # Load the tensor
        image_tensor = TF.resize(image_tensor, self.target_size)  # Resize
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, torch.tensor(label, dtype=torch.long)

# CNN Model
class MammogramCNN(nn.Module):
    def __init__(self):
        super(MammogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flattened_size = 64 * (256 // 4) * (256 // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

# Load full test dataset
full_test_dataset = MammogramDataset(
    "datasets/tensor_dataset_mias_test/",
    "datasets/test_labels.csv",
    transform=transform
)

# Randomly select 30 images
random_indices = random.sample(range(len(full_test_dataset)), 30)
subset_test_dataset = Subset(full_test_dataset, random_indices)
test_loader = DataLoader(subset_test_dataset, batch_size=16, shuffle=False)

# Load model
model = MammogramCNN().to(device)
model.load_state_dict(torch.load("mammogram_cnn_mias.pth"))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy on 30 Random Images: {100 * correct / total:.2f}%")
