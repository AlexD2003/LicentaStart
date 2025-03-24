import pandas as pd

# Load the original labels file
df = pd.read_csv("labels.csv")

# Convert abnormality column to binary labels (0 = Normal, 1 = Abnormal)
df["label"] = df["abnormality"].apply(lambda x: 0 if x == "NORM" else 1)

# Keep only 'image_id' and 'label'
df = df[["image_id", "label"]]

# Save the modified CSV
df.to_csv("modified_labels.csv", index=False)

print("Modified labels saved as 'modified_labels.csv'")
