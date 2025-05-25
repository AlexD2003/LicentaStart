import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MammogramDataset(Dataset):
    def __init__(self, tensor_folder, label_file, transform=None, target_size=(224, 224)):
        self.tensor_folder = tensor_folder
        self.labels_df = pd.read_csv(label_file, dtype={"image_id": str, "label": int})
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        file_name = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])
        tensor_path = os.path.join(self.tensor_folder, file_name + ".pt")
        image_tensor = torch.load(tensor_path)
        image_tensor = TF.resize(image_tensor, self.target_size)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)
