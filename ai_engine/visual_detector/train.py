import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import PayslipDataset
from model import build_model

IMAGE_DIR = "dataset_processed/images"
MASK_DIR = "dataset_processed/masks"
ELA_DIR = "dataset_processed/ela"

BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = PayslipDataset(IMAGE_DIR, MASK_DIR, ELA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


model = build_model().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for i, (images, masks) in enumerate(dataloader):
        print("Batch", i)

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


torch.save(model.state_dict(), "tamper_model.pth")

print("Model training complete.")