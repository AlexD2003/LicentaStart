import cv2
import numpy as np
import os

# Input and Output directories
input_folder = "dataset/"
output_folder = "cleaned_dataset/"

os.makedirs(output_folder, exist_ok=True)

def remove_artifacts(image_path, output_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error loading {image_path}")
        return

    # Resize to 1024x1024 for consistency
    img = cv2.resize(img, (1024, 1024))

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu’s thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in {image_path}. Skipping...")
        return

    # Create a mask with only the largest contour (assumed to be the breast)
    mask = np.zeros_like(binary)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    cleaned = cv2.bitwise_and(img, img, mask=mask)

    # Remove pectoral muscle (top-left region)
    height, width = cleaned.shape
    triangle = np.array([[0, 0], [int(width * 0.3), 0], [0, int(height * 0.3)]], np.int32)
    cv2.fillPoly(mask, [triangle], 0)  # Black-out the pectoral region
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=mask)

    # Auto-crop to remove excess black areas
    coords = cv2.findNonZero(mask)  # Find all non-zero points
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
        cleaned = cleaned[y:y+h, x:x+w]  # Crop to content

    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cleaned = clahe.apply(cleaned)

    # Save as 8-bit grayscale (force no transparency)
    cv2.imwrite(output_path, cleaned)

# Process all images
for file_name in os.listdir(input_folder):
    if file_name.endswith(".pgm"):  # Process only .pgm files
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        remove_artifacts(input_path, output_path)

print("Cleaning complete! Images saved in", output_folder)
