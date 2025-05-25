import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_FOLDER = "datasets/training_mass_selected"
OUTPUT_FOLDER = "datasets/processed_png"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize
    img = cv2.resize(img, (1024, 1024))

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find largest contour (breast area)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Apply mask
    cleaned = cv2.bitwise_and(img, img, mask=mask)

    # Remove pectoral muscle (upper-left triangle)
    height, width = cleaned.shape
    triangle = np.array([[0, 0], [int(width * 0.3), 0], [0, int(height * 0.3)]], np.int32)
    cv2.fillPoly(mask, [triangle], 0)
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=mask)

    # Crop to non-zero region
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cleaned = cleaned[y:y+h, x:x+w]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cleaned = clahe.apply(cleaned)

    return cleaned

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".png")]
    for fname in tqdm(files, desc="🧪 Preprocessing"):
        input_path = os.path.join(INPUT_FOLDER, fname)
        output_path = os.path.join(OUTPUT_FOLDER, fname)
        try:
            processed = preprocess_image(input_path)
            cv2.imwrite(output_path, processed)
        except Exception as e:
            print(f"❌ Failed on {fname}: {e}")
    print(f"\n✅ Finished full preprocessing of {len(files)} images.")

if __name__ == "__main__":
    main()
