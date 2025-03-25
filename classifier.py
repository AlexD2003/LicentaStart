import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import os
import pandas as pd

# Define a custom dataset

class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, label_file, transform=None, target_size=(256, 256)):
        self.tensor_folder = tensor_folder
        self.labels_df = pd.read_csv(label_file, dtype={"image_id": str, "label": int})

        self.transform = transform
        self.target_size = target_size  # Set the target size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        file_name = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])
        tensor_path = os.path.join(self.tensor_folder, file_name + ".pt")  # Ensure correct file extension
        image_tensor = torch.load(tensor_path)  # No need to unsqueeze here, just load the tensor

        # Resize the image to (1, target_size[0], target_size[1])
        image_tensor = TF.resize(image_tensor, self.target_size)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)


# Define a simple CNN model
class MammogramCNN(nn.Module):
    def __init__(self):
        super(MammogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the flattened size dynamically
        self.flattened_size = 64 * (256 // 4) * (256 // 4)  # Two max pools reduce by 4x
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Dynamically flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
dataset = MammogramDataset("tensor_dataset/", "modified_labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cpu")
model = MammogramCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training complete!")
torch.save(model.state_dict(), "mammogram_cnn.pth")
