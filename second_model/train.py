import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.model import ResNet18Classifier
from utils.dataset import MammogramDataset
import os

# ======= Config =======
DATA_DIR = "datasets/tensor_dataset_mias/"
LABEL_CSV = "datasets/modified_labels.csv"
BATCH_SIZE = 16
EPOCHS = 15  # Increased a bit to help fine-tuning
LEARNING_RATE = 1e-5  # Lower LR for fine-tuning
MODEL_PATH = "resnet18_mammogram.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Transforms =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,))
])

# ======= Dataset & Loader =======
dataset = MammogramDataset(DATA_DIR, LABEL_CSV, transform=transform, target_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======= Model =======
model = ResNet18Classifier(pretrained=True).to(DEVICE)

# UNFREEZE all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: halve LR every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ======= Training Loop =======
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()  # Adjust learning rate

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# ======= Save Model =======
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Training complete! Model saved to {MODEL_PATH}")
