import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

# Define a custom dataset
class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, transform=None):
        self.tensor_folder = tensor_folder
        self.tensor_files = [f for f in os.listdir(tensor_folder) if f.endswith(".pt")]
        self.transform = transform

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_folder, self.tensor_files[idx])
        image_tensor = torch.load(tensor_path)  # Load tensor
        label = 1 if "cancer" in tensor_path else 0  # Simple labeling (modify as needed)

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
        self.fc1 = nn.Linear(64 * 256 * 256, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (cancer vs. no cancer)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
dataset = MammogramDataset("tensor_dataset/", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MammogramCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
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
