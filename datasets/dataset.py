import torch
from torch.utils.data import Dataset, DataLoader
import os

class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, transform=None):
        self.tensor_folder = tensor_folder
        self.file_names = [f for f in os.listdir(tensor_folder) if f.endswith('.pt')]  # Load only tensor files
        self.transform = transform  # Optional transformations
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_folder, self.file_names[idx])
        image_tensor = torch.load(file_path)  # Load tensor image

        # Extract label from filename (example: 'positive_01.pt' → label=1)
        label = 1 if "positive" in self.file_names[idx] else 0  # Adjust according to dataset

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)

# Set folder path
tensor_folder = "tensor_dataset/"

# Create dataset instance
dataset = MammogramDataset(tensor_folder)

# Test dataset
image, label = dataset[0]
print("Image Shape:", image.shape)  # Should be (1, 1024, 1024)
print("Label:", label)
