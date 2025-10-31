from ultralytics import YOLO
import cv2
from utils.preprocessing import preprocess_yolov8


def load_model(model_variant="runs/detect/train/weights/best.pt"):
    from ultralytics import YOLO
    model = YOLO(model_variant)
    return model

def detect(model, image_source):
    """
    Runs inference using YOLOv8 model.
    Returns Ultralytics Results list.
    """
    image_input = preprocess_yolov8(image_source)
    results = model(image_input)
    return results
