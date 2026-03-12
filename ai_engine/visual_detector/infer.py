import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from model import build_model


# ---------- CONFIG ----------
MODEL_PATH = "tamper_model.pth"
IMAGE_PATH = "test_image.tif"
IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ---------- PREPROCESS IMAGE ----------
def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)

    return img, img_tensor


# ---------- RUN INFERENCE ----------
def detect_tampering(image_path):

    original_img, img_tensor = preprocess_image(image_path)

    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)["out"]

    mask = torch.sigmoid(output)
    mask = mask.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255

    return original_img, mask


# ---------- HEATMAP OVERLAY ----------
def create_heatmap(original, mask):

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
        0.7,
        heatmap,
        0.3,
        0
    )

    return overlay


# ---------- MAIN ----------
if __name__ == "__main__":

    original, mask = detect_tampering(IMAGE_PATH)

    overlay = create_heatmap(original, mask)

    cv2.imwrite("tamper_mask.png", mask)
    cv2.imwrite("tamper_heatmap.png", overlay)

    print("Tamper detection complete.")
    print("Saved: tamper_mask.png and tamper_heatmap.png")