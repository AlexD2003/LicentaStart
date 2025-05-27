import os
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIG ===
CSV_PATH = "csvs/mias_test.csv"
IMAGE_DIR = "datasets/images_test"
MODEL_PATH = "models/fasterrcnn_mias.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.1

# === Load ground truth labels
labels_df = pd.read_csv(CSV_PATH).dropna(subset=["label"])
labels_df["id"] = labels_df["id"].str.strip()
labels_df["label"] = labels_df["label"].map({"B": 0, "M": 1})
gt_map = dict(zip(labels_df["id"], labels_df["label"]))

# === Load model with matching head
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# === Evaluation loop
y_true, y_pred = [], []

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".pgm")])
print(f"🔍 Evaluating on {len(image_files)} test images...")

for file in image_files:
    image_id = os.path.splitext(file)[0]
    gt_label = gt_map.get(image_id)
    if gt_label is None:
        continue  # skip unannotated

    image_path = os.path.join(IMAGE_DIR, file)
    image = Image.open(image_path).convert("L")
    tensor_img = F.to_tensor(image).repeat(3, 1, 1).to(DEVICE)

    with torch.no_grad():
        outputs = model([tensor_img])[0]

    scores = outputs['scores'].cpu()
    labels = outputs['labels'].cpu()

    pred_label = -1
    for score, label in zip(scores, labels):
        if score >= SCORE_THRESHOLD:
            pred_label = label.item()
            break

    if pred_label == -1:
        pred_label = 0  # assume negative if nothing detected

    y_true.append(gt_label)
    y_pred.append(pred_label)
    print(f"🖼️ {file}: GT={gt_label}, Pred={pred_label}")

# === Results
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
