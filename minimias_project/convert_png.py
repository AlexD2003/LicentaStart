from pathlib import Path
import cv2

# Paths
input_dir = Path("datasets/images_test")  # adjust if needed
output_dir = Path("datasets/images_test_png")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert all .pgm files
pgms = list(input_dir.glob("*.pgm"))
if not pgms:
    print("❌ No .pgm files found in", input_dir.resolve())
else:
    for pgm_path in pgms:
        img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Failed to read {pgm_path.name}")
            continue
        out_path = output_dir / (pgm_path.stem + ".png")
        cv2.imwrite(str(out_path), img)
        print(f"✅ {pgm_path.name} → {out_path.name}")

    print(f"\n🎉 Done! Converted {len(pgms)} images to {output_dir.resolve()}")
