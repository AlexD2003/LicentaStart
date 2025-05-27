import os
import shutil
import pandas as pd

# === CONFIG ===
IMAGE_DIR = "datasets/raw"  # where all .pgm files currently are
TRAIN_CSV = "csvs/mias_train.csv"
TEST_CSV = "csvs/mias_test.csv"
TRAIN_DIR = "datasets/images_train"
TEST_DIR = "datasets/images_test"

# === Create output dirs
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# === Load image IDs from CSVs
train_ids = pd.read_csv(TRAIN_CSV)["id"].unique()
test_ids = pd.read_csv(TEST_CSV)["id"].unique()

def copy_images(id_list, dest_dir):
    for image_id in id_list:
        filename = image_id + ".pgm"
        src_path = os.path.join(IMAGE_DIR, filename)
        dst_path = os.path.join(dest_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"⚠️ Missing: {src_path}")

copy_images(train_ids, TRAIN_DIR)
copy_images(test_ids, TEST_DIR)

print(f"✅ Copied {len(train_ids)} images to {TRAIN_DIR}")
print(f"✅ Copied {len(test_ids)} images to {TEST_DIR}")
