import os
import pandas as pd

# === CONFIG ===
CSV_PATH = "csvs/mias_labels.csv"
YOLO_LABEL_DIR = "datasets/yolo_labels"
IMG_DIR = "datasets/images_raw"  # used to get image size for normalization
IMAGE_SIZE = (1024, 1024)  # update if you're resizing the .pgm images

os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["x", "y", "radius"])  # only lesions

# Class map: benign = 0, malignant = 1
def label_to_class(label):
    return 0 if label == "B" else 1

# === Convert each ROI
id_counts = {}

for _, row in df.iterrows():
    image_id = row["id"]
    label = row["label"]
    x, y, r = float(row["x"]), float(row["y"]), float(row["radius"])
    
    # Calculate box
    img_w, img_h = IMAGE_SIZE
    xc, yc = x / img_w, y / img_h
    w, h = (2 * r) / img_w, (2 * r) / img_h
    
    # YOLO format: class_id xc yc w h
    class_id = label_to_class(label)
    line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
    
    count = id_counts.get(image_id, 0)
    label_path = os.path.join(YOLO_LABEL_DIR, f"{image_id}.txt")
    
    with open(label_path, "a") as f:
        f.write(line)
    
    id_counts[image_id] = count + 1

print(f"✅ YOLOv5 annotations saved to: {YOLO_LABEL_DIR}")
print(f"💡 Total labeled images: {len(id_counts)}")
