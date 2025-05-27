import os
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

# === CONFIG ===
input_dirs = {
    "train": "datasets/images_train",
    "test": "datasets/images_test"
}
csv_paths = {
    "train": "csvs/mias_train.csv",
    "test": "csvs/mias_test.csv"
}
output_base = Path("datasets/tensor_fullimage")
img_size = (240, 240)

# === Ensure output dirs
for split in ["train", "test"]:
    (output_base / split).mkdir(parents=True, exist_ok=True)

# === Image Processing Pipeline
def enhance_image(img):
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Optional denoise with slight blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    # Stretch histogram
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img

# === Preprocess split
def process_split(split):
    df = pd.read_csv(csv_paths[split])
    count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        img_id = row["id"]
        label = 1 if row["label"] == "M" else 0
        img_path = Path(input_dirs[split]) / f"{img_id}.pgm"
        out_path = output_base / split / f"{img_id}.pt"

        if not img_path.exists():
            print(f"❌ Missing: {img_path}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Failed to read: {img_path}")
            continue

        img = enhance_image(img)
        img = cv2.resize(img, img_size)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # shape: 1 x H x W

        sample = {
            "tensor": tensor,
            "label": label
        }

        torch.save(sample, out_path)
        count += 1

    return count

# === Run both splits
if __name__ == "__main__":
    train_count = process_split("train")
    test_count = process_split("test")
    print(f"\n✅ Done. Saved {train_count} train and {test_count} test tensors.")
