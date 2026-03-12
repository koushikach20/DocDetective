import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths
INPUT_DIR = "dataset_processed/images"
OUTPUT_DIR = "dataset_processed/ela"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_ela(image_path, output_path, quality=90):

    # Load image
    original = Image.open(image_path).convert("RGB")

    # Temporary recompressed image
    temp_path = "temp.jpg"
    original.save(temp_path, "JPEG", quality=quality)

    recompressed = Image.open(temp_path)

    # Convert to numpy
    original_np = np.array(original)
    recompressed_np = np.array(recompressed)

    # Compute difference
    ela = np.abs(original_np.astype(int) - recompressed_np.astype(int))

    # Scale differences for visibility
    max_diff = ela.max()
    if max_diff == 0:
        max_diff = 1

    ela = (ela * (255.0 / max_diff)).astype(np.uint8)

    # Save ELA image
    ela_img = Image.fromarray(ela)
    ela_img.save(output_path)


def process_dataset():

    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]

    for img_name in tqdm(images, desc="Generating ELA"):

        img_path = os.path.join(INPUT_DIR, img_name)
        ela_path = os.path.join(OUTPUT_DIR, img_name)

        generate_ela(img_path, ela_path)


if __name__ == "__main__":
    process_dataset()