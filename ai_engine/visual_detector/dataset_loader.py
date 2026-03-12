import os
import cv2
import torch
from torch.utils.data import Dataset


class PayslipDataset(Dataset):

    def __init__(self, image_dir, mask_dir, ela_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ela_dir = ela_dir

        self.images = sorted(os.listdir(image_dir))[:80]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        ela_path = os.path.join(self.ela_dir, img_name)

        image = cv2.imread(img_path)
        ela = cv2.imread(ela_path)

        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (512, 512))
        ela = cv2.resize(ela, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        image = image / 255.0
        ela = ela / 255.0

        combined = cv2.addWeighted(image, 0.7, ela, 0.3, 0)

        combined = torch.tensor(combined).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0

        return combined, mask