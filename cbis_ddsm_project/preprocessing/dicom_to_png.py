import os
import pydicom
import numpy as np
from PIL import Image

RAW_DIR = "datasets/raw"
OUTPUT_DIR = "datasets/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_image(img_array):
    img_array = img_array.astype(np.float32)
    img_array -= np.min(img_array)
    img_array /= (np.max(img_array) + 1e-8)
    img_array *= 255.0
    return img_array.astype(np.uint8)

def convert_dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    img_array = ds.pixel_array
    img_normalized = normalize_image(img_array)
    img = Image.fromarray(img_normalized)
    img.save(output_path)

def is_full_mammogram_dir(path):
    return "full mammogram images" in path

def main():
    print(f"📂 Scanning {RAW_DIR} for full mammogram DICOMs...")
    count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        if is_full_mammogram_dir(root):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dicom_path = os.path.join(root, file)
                    study_id = root.split("/")[-3]
                    view_type = root.split("/")[-2]
                    output_name = f"{study_id}_{view_type}_{file.replace('.dcm', '')}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_name)
                    try:
                        convert_dicom_to_png(dicom_path, output_path)
                        os.remove(dicom_path)
                        print(f"✅ Converted and deleted: {dicom_path}")
                        count += 1
                    except Exception as e:
                        print(f"❌ Failed to convert {dicom_path}: {e}")
    print(f"\n✅ Finished! Total converted: {count}")

if __name__ == "__main__":
    main()
