import os
import cv2
import numpy as np

INPUT_FOLDER = "datasets/testing_mass_selected"
OUTPUT_FOLDER = "datasets/cleaned_test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load {image_path}")
        return

    img = cv2.resize(img, (1024, 1024))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No contours in {image_path}")
        return

    mask = np.zeros_like(binary)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((10, 10), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    height, width = img.shape
    triangle = np.array([[0, 0], [int(width * 0.4), 0], [0, int(height * 0.3)]], np.int32)
    cv2.fillPoly(mask, [triangle], 0)
    cleaned = cv2.bitwise_and(img, img, mask=mask)

    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cleaned = cleaned[y:y+h, x:x+w]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cleaned = clahe.apply(cleaned)

    cv2.imwrite(output_path, cleaned)

for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".png"):
        inp = os.path.join(INPUT_FOLDER, fname)
        out = os.path.join(OUTPUT_FOLDER, fname)
        preprocess_image(inp, out)

print("✅ All test images preprocessed and saved.")
