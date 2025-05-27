import os
import torch
from PIL import Image
import torchvision.transforms as transforms

INPUT_FOLDER = "datasets/cleaned_test"
OUTPUT_FOLDER = "datasets/tensor_dataset_test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".png"):
        img_path = os.path.join(INPUT_FOLDER, fname)
        out_path = os.path.join(OUTPUT_FOLDER, fname.replace(".png", ".pt"))
        img = Image.open(img_path).convert("L")
        tensor = transform(img)
        torch.save(tensor, out_path)
        print(f"✅ Saved {fname} -> .pt")

print("✅ All test images converted to tensors.")
