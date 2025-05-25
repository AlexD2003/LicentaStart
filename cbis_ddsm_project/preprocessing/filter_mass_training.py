import os
import shutil

SOURCE_DIR = "datasets/cleaned"
DEST_DIR = "datasets/training_mass_selected"

os.makedirs(DEST_DIR, exist_ok=True)

count = 0
for file_name in os.listdir(SOURCE_DIR):
    if "Mass-Training" in file_name and file_name.lower().endswith(".png"):
        src_path = os.path.join(SOURCE_DIR, file_name)
        dst_path = os.path.join(DEST_DIR, file_name)
        try:
            shutil.copy2(src_path, dst_path)
            count += 1
        except Exception as e:
            print(f"❌ Failed to copy {file_name}: {e}")

print(f"✅ Copied {count} Mass-Training PNGs to {DEST_DIR}")
