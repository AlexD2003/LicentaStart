import os
import pandas as pd
import torch
import pydicom
import numpy as np
from torchvision.transforms.functional import resize
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "csvs/mass_case_description_train_set.csv"
TENSOR_INPUT_DIR = "datasets/tensor_dataset"
MASK_BASE_DIR = "/mnt/ssd/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/training_mass_selected"  # Adjust if different
OUTPUT_DIR = "datasets/tensor_dataset_roi"
TARGET_SIZE = (240, 240)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
df["image file path"] = df["image file path"].str.strip('"')
df["ROI mask file path"] = df["ROI mask file path"].str.strip('"')

def extract_image_id(path):
    return os.path.splitext(os.path.basename(path))[0]

df["image_id"] = df["image file path"].apply(extract_image_id)

# === Process each image ===
skipped = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row["image_id"]
    tensor_path = os.path.join(TENSOR_INPUT_DIR, image_id + ".pt")
    output_path = os.path.join(OUTPUT_DIR, image_id + ".pt")
    
    mask_rel_path = row["ROI mask file path"]
    mask_dcm_path = os.path.join(MASK_BASE_DIR, mask_rel_path)

    if not os.path.exists(tensor_path) or not os.path.exists(mask_dcm_path):
        skipped += 1
        continue

    try:
        tensor = torch.load(tensor_path)  # shape [1, H, W]
        mask_dcm = pydicom.dcmread(mask_dcm_path)
        mask_array = mask_dcm.pixel_array  # shape [H, W]

        # Compute bounding box from mask
        mask = (mask_array > 0).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            skipped += 1
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        cropped = tensor[:, y1:y2+1, x1:x2+1]
        resized = resize(cropped, TARGET_SIZE)
        torch.save(resized, output_path)
    except Exception as e:
        print(f"❌ Failed on {image_id}: {e}")
        skipped += 1

print(f"\n✅ Done. Skipped {skipped} samples due to missing data or invalid masks.")
