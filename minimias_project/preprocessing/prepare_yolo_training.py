import os
import shutil
import random
import cv2
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
IMAGE_DIR = "datasets/raw"
LABEL_DIR = "datasets/yolo_labels"
YOLO_ROOT = "datasets/yolo_dataset"
IMG_SIZE = (1024, 1024)
VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)

# === OUTPUT DIRS
paths = {
    "train/images": os.path.join(YOLO_ROOT, "train/images"),
    "train/labels": os.path.join(YOLO_ROOT, "train/labels"),
    "val/images": os.path.join(YOLO_ROOT, "val/images"),
    "val/labels": os.path.join(YOLO_ROOT, "val/labels"),
}

for path in paths.values():
    os.makedirs(path, exist_ok=True)

# === Get labeled image list
labeled_images = [f.replace(".txt", "") for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]
random.shuffle(labeled_images)
split_idx = int(len(labeled_images) * (1 - VAL_SPLIT))
train_ids = labeled_images[:split_idx]
val_ids = labeled_images[split_idx:]

def process_and_copy(image_id, split):
    src_img = os.path.join(IMAGE_DIR, image_id + ".pgm")
    dst_img = os.path.join(YOLO_ROOT, split, "images", image_id + ".jpg")
    src_lbl = os.path.join(LABEL_DIR, image_id + ".txt")
    dst_lbl = os.path.join(YOLO_ROOT, split, "labels", image_id + ".txt")

    # === Convert PGM to JPG (YOLO prefers JPG/PNG)
    img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Skipped: {src_img} not found or unreadable.")
        return
    img_resized = cv2.resize(img, IMG_SIZE)
    cv2.imwrite(dst_img, img_resized)

    # === Copy label
    shutil.copy2(src_lbl, dst_lbl)

# === Process sets
for image_id in tqdm(train_ids, desc="Train Set"):
    process_and_copy(image_id, "train")

for image_id in tqdm(val_ids, desc="Val Set"):
    process_and_copy(image_id, "val")

print(f"\n✅ YOLOv5 dataset ready at: {YOLO_ROOT}")
print(f"📸 Train: {len(train_ids)} images | 🧪 Val: {len(val_ids)} images")
