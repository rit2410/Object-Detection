"""
rt_detr.py
-----------
RT-DETR (Real-Time DETR) via Ultralytics API.
Transformer-based object detector.
"""

from ultralytics import RTDETR
from utils.preprocessing import preprocess_yolov8

def load_model(model_variant="rtdetr-l.pt"):
    """
    Loads pretrained RT-DETR model from Ultralytics.
    Variants: rtdetr-l.pt, rtdetr-x.pt
    """
    model = RTDETR(model_variant)
    return model

def detect(model, image_source):
    """
    Runs inference using RT-DETR model.
    Returns Ultralytics Results list (same format as YOLOv8).
    """
    image_input = preprocess_yolov8(image_source)
    results = model(image_input)
    return results
