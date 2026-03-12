import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_model(num_classes=1):

    model = deeplabv3_resnet50(pretrained=True)

    # Replace classifier layer
    model.classifier[4] = nn.Conv2d(
        256,
        num_classes,
        kernel_size=1
    )

    return model