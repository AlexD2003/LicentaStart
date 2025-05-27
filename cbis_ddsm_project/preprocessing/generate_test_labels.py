import os
import pandas as pd
import re
import csv

# Paths
IMAGE_FOLDER = "datasets/testing_mass_selected"
METADATA_FILE = "csvs/mass_case_description_test_set.csv"
OUTPUT_CSV = "csvs/test_labels.csv"

# Load metadata
meta = pd.read_csv(METADATA_FILE)

# Extract image folder name from metadata
meta["image_id"] = meta["cropped image file path"].apply(lambda x: x.split("/")[0].replace('"', '').strip())
meta["label"] = meta["pathology"].apply(lambda p: 1 if str(p).strip().upper() == "MALIGNANT" else 0)
label_dict = dict(zip(meta["image_id"], meta["label"]))

# Match actual PNG files
entries = []
for fname in os.listdir(IMAGE_FOLDER):
    if not fname.lower().endswith(".png"):
        continue
    fname = fname.replace('"', '').strip()

    # This regex extracts: Mass-Test_P_XXXXX_LEFT_CC_1 from the filename
    match = re.match(r'^("?Mass-Test_P_\d+_[A-Z]+_[A-Z]+)_.*?_1-1\.png$', fname)
    if match:
        prefix = match.group(1).replace('"', '') + "_1"
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
