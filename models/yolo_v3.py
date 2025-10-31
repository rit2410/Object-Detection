"""
yolo_v3.py
-----------
Baseline CNN-style detector using Faster R-CNN (ResNet-50 FPN backbone).
Serves as YOLOv3-equivalent classical baseline.
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from utils.preprocessing import load_image, preprocess_torch

def load_model(device):
    """Loads pretrained Faster R-CNN (COCO) model."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model, weights.transforms()

def detect(model, image_source, device):
    """Runs inference and returns dict of boxes, labels, scores."""
    image = load_image(image_source)
    image_tensor = preprocess_torch(image, device)
    with torch.no_grad():
        preds = model(image_tensor)[0]
    return preds
