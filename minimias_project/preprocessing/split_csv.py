import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === CONFIG ===
INPUT_CSV = "csvs/mias_labels.csv"
TRAIN_CSV = "csvs/mias_train.csv"
TEST_CSV = "csvs/mias_test.csv"
TEST_RATIO = 0.2  # 20% for testing

# === Load and prep ===
df = pd.read_csv(INPUT_CSV)

# Determine label per image (based on any abnormality with a label)
def label_type(subdf):
    if subdf["label"].isin(["M"]).any():
        return "M"
    elif subdf["label"].isin(["B"]).any():
        return "B"
    return "NORM"

grouped = df.groupby("id").apply(label_type).reset_index()
grouped.columns = ["id", "label"]
strat_labels = grouped["label"]

# === Train-test split by image ID ===
train_ids, test_ids = train_test_split(
    grouped["id"], test_size=TEST_RATIO, stratify=strat_labels, random_state=42
)

# === Filter original CSV ===
train_df = df[df["id"].isin(train_ids)]
test_df = df[df["id"].isin(test_ids)]

# === Save ===
os.makedirs("csvs", exist_ok=True)
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"✅ Train set: {len(train_df)} entries")
print(f"✅ Test set:  {len(test_df)} entries")
print(f"📝 Saved to: {TRAIN_CSV}, {TEST_CSV}")
