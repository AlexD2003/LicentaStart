import cv2
import numpy as np
import os

input_folder = "dataset/"
output_folder = "outputProcessing/"

os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image_path, output_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 1024x1024
    img = cv2.resize(img, (1024, 1024))

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Noise reduction with Gaussian Blur
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Save preprocessed image
    cv2.imwrite(output_path, img)

# Loop through images in the dataset folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".pgm"):  # Process only .pgm files
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        preprocess_image(input_path, output_path)

print("Preprocessing complete! Images saved in", output_folder)
