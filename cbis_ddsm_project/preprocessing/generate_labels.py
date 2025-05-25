import os
import pandas as pd
import re
import csv

IMAGE_FOLDER = "datasets/training_mass_selected"
METADATA_FILE = "csvs/mass_case_description_train_set.csv"
OUTPUT_CSV = "csvs/train_labels.csv"

# Load and parse metadata
meta = pd.read_csv(METADATA_FILE)

# Extract base folder name from image file path column
meta["image_id"] = meta["image file path"].apply(lambda x: x.split("/")[0].strip().replace('"', ''))

# Convert pathology to binary label
meta["label"] = meta["pathology"].apply(lambda p: 1 if str(p).strip().upper() == "MALIGNANT" else 0)

# Create lookup dictionary
label_dict = dict(zip(meta["image_id"], meta["label"]))

# Match actual PNGs to metadata
entries = []
for fname in os.listdir(IMAGE_FOLDER):
    if not fname.lower().endswith(".png"):
        continue
    fname = fname.replace('"', '')  # Clean up weird quoting

    # Extract something like Mass-Training_P_01386_LEFT_CC from filename
    match = re.match(r'^([^_]+_P_\d+_[A-Z]+_[A-Z]+)', fname)
    if match:
        prefix = match.group(1)
        if prefix in label_dict:
            entries.append({"image_id": fname, "label": label_dict[prefix]})
        else:
            print(f"⚠️ No match in metadata: {prefix} from {fname}")
    else:
        print(f"⚠️ Unable to parse filename: {fname}")

# Save to CSV
df = pd.DataFrame(entries)
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"\n✅ Labels written to {OUTPUT_CSV} — {len(df)} entries.")
