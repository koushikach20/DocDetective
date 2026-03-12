import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_ROOT = "dataset/Payslip_dataset_shared"
OUTPUT_ROOT = "dataset_processed"

IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")
MASK_DIR = os.path.join(OUTPUT_ROOT, "masks")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


def convert_tif_to_png(input_path, output_path):
    img = cv2.imread(input_path)
    cv2.imwrite(output_path, img)


def create_empty_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(mask_path, mask)


def process_genuine():
    genuine_path = os.path.join(DATASET_ROOT, "DOC_Genuine")

    files = [f for f in os.listdir(genuine_path) if f.endswith(".tif")]

    for f in tqdm(files, desc="Processing Genuine"):

        src_img = os.path.join(genuine_path, f)

        img_name = f.replace(".tif", ".png")

        dst_img = os.path.join(IMAGE_DIR, img_name)
        dst_mask = os.path.join(MASK_DIR, img_name)

        convert_tif_to_png(src_img, dst_img)
        create_empty_mask(dst_img, dst_mask)


def process_forged():

    forged_root = os.path.join(DATASET_ROOT, "DOC_Forged")

    for root, dirs, files in os.walk(forged_root):

        tif_files = [f for f in files if f.endswith(".tif")]

        for tif_file in tqdm(tif_files, desc="Processing Forged"):

            prefix = tif_file.replace(".tif", "")

            img_path = os.path.join(root, tif_file)
            mask_path = os.path.join(root, prefix + ".word.png")

            if not os.path.exists(mask_path):
                continue

            img_name = prefix + ".png"

            dst_img = os.path.join(IMAGE_DIR, img_name)
            dst_mask = os.path.join(MASK_DIR, img_name)

            convert_tif_to_png(img_path, dst_img)

            mask = cv2.imread(mask_path, 0)
            cv2.imwrite(dst_mask, mask)


def main():

    process_genuine()
    process_forged()

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()