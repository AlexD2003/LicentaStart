import os
import shutil

RAW_DIR = "datasets/cleaned"  # Where all PNGs live
DEST_DIR = "datasets/testing_mass_selected"

os.makedirs(DEST_DIR, exist_ok=True)

count = 0
for fname in os.listdir(RAW_DIR):
    if "Mass-Test" in fname and fname.lower().endswith(".png"):
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(DEST_DIR, fname)
        try:
            shutil.copy2(src, dst)
            count += 1
        except Exception as e:
            print(f"❌ Failed to copy {fname}: {e}")

print(f"\n✅ Copied {count} Mass-Test PNGs to {DEST_DIR}")
