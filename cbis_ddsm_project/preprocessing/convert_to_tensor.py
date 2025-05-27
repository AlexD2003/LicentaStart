import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

INPUT_FOLDER = "datasets/processed_png"
OUTPUT_FOLDER = "datasets/tensor_dataset"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def image_to_tensor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img /= 255.0  # Normalize to 0–1
    tensor = torch.from_numpy(img).unsqueeze(0)  # Add channel dim: [1, H, W]
    return tensor

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".png")]
    for fname in tqdm(files, desc="📦 Converting to .pt"):
        input_path = os.path.join(INPUT_FOLDER, fname)
        output_name = fname.replace('"', '').replace(".png", ".pt")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        try:
            tensor = image_to_tensor(input_path)
            torch.save(tensor, output_path)
        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")
    print(f"\n✅ Saved {len(files)} image tensors to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
