import os
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# === CONFIG ===
CSV_PATH = "csvs/mias_train.csv"
IMAGE_DIR = "datasets/images_train"
MODEL_PATH = "models/fasterrcnn_mias.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 10

# === Custom Dataset ===
class MIASDetectionDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path).dropna(subset=["x", "y", "radius"])
        self.df["id"] = self.df["id"].str.strip()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]
        label = 1 if row["label"] == "M" else 0

        img_path = os.path.join(self.image_dir, f"{image_id}.pgm")
        image = Image.open(img_path).convert("L")
        image = F.to_tensor(image).repeat(3, 1, 1)  # grayscale → fake RGB

        # Convert (x, y, r) to [x1, y1, x2, y2]
        cx, cy, r = float(row["x"]), float(row["y"]), float(row["radius"])
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        print(f"🖼️ Loaded: {image_id}.pgm | Label: {'Malignant' if label else 'Benign'} | Box: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
        return image, target

# === Dataset Split ===
dataset = MIASDetectionDataset(CSV_PATH, IMAGE_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === Load Faster R-CNN with pretrained weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# Replace classifier for 2-class output
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

model.to(DEVICE)

# === Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005)

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"   📦 Batch {batch_idx+1:02d} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} complete | Total Loss: {epoch_loss:.4f}")

# === Save Model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
