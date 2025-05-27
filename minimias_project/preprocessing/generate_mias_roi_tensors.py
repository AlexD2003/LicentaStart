import os
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

# === CONFIG ===
TRAIN_CSV = "csvs/mias_train.csv"
TEST_CSV = "csvs/mias_test.csv"
TRAIN_IMG_DIR = "datasets/images_train"
TEST_IMG_DIR = "datasets/images_test"
TRAIN_OUT_DIR = "datasets/tensor_dataset_roi/train"
TEST_OUT_DIR = "datasets/tensor_dataset_roi/test"
TARGET_SIZE = (240, 240)

# === Create output directories
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
os.makedirs(TEST_OUT_DIR, exist_ok=True)

# === CLAHE enhancer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# === Utility: load .pgm and apply CLAHE
def load_and_preprocess_pgm(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    enhanced = clahe.apply(img)
    tensor = torch.tensor(enhanced, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
    return tensor

# === Utility: crop ROI from tensor
def crop_roi(tensor, x, y, radius):
    h, w = tensor.shape[1], tensor.shape[2]
    r = int(radius)
    x, y = int(x), int(y)
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)
    return tensor[:, y1:y2, x1:x2]

# === Process dataset
def process_set(csv_path, image_dir, output_dir):
    df = pd.read_csv(csv_path)
    id_counts = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["id"]
        label = row["label"]

        if pd.isna(row["x"]) or pd.isna(row["y"]) or pd.isna(row["radius"]):
            continue  # skip normal images

        x, y, r = row["x"], row["y"], row["radius"]
        filename = image_id + ".pgm"
        img_path = os.path.join(image_dir, filename)

        if not os.path.exists(img_path):
            print(f"⚠️ Missing image: {img_path}")
            continue

        try:
            tensor = load_and_preprocess_pgm(img_path)
            roi_tensor = crop_roi(tensor, x, y, r)
            resized_tensor = resize(roi_tensor, TARGET_SIZE)

            # handle multiple lesions per image
            count = id_counts.get(image_id, 0)
            out_name = f"{image_id}_{count}.pt"
            id_counts[image_id] = count + 1

            torch.save(resized_tensor, os.path.join(output_dir, out_name))
        except Exception as e:
            print(f"❌ Failed on {image_id}: {e}")

# === Run both sets
process_set(TRAIN_CSV, TRAIN_IMG_DIR, TRAIN_OUT_DIR)
process_set(TEST_CSV, TEST_IMG_DIR, TEST_OUT_DIR)

print(f"\n✅ Conversion complete. Tensors saved to:")
print(f"- {TRAIN_OUT_DIR}")
print(f"- {TEST_OUT_DIR}")
