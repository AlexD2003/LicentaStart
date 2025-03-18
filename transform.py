import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Paths
input_folder = "cleaned_dataset/"
output_tensor_folder = "tensor_dataset/"

# Create output folder if it doesn’t exist
os.makedirs(output_tensor_folder, exist_ok=True)

# Define transformation: Convert to Tensor & Normalize (0-1)
transform = transforms.Compose([
    transforms.ToTensor()  # Converts image to tensor and scales 0-255 to 0-1
])

def process_image(image_path, output_path):
    # Load image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Apply transformation
    tensor_img = transform(img)

    # Save tensor (Torch saves tensors in .pt format)
    torch.save(tensor_img, output_path)

# Process all images in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".pgm"):  # Only process PGM images
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_tensor_folder, file_name.replace(".pgm", ".pt"))
        process_image(input_path, output_path)

print("✅ All images converted to tensors and saved in:", output_tensor_folder)
