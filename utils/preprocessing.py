import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    import cv2
except Exception as e:
    print(f"[WARNING] OpenCV failed to load: {e}")
    import types
    cv2 = types.SimpleNamespace()
    for fn in ["imread", "imwrite", "resize", "imshow", "cvtColor"]:
        setattr(cv2, fn, lambda *a, **k: None)
    cv2.IMREAD_COLOR = 1

import numpy as np
from PIL import Image
import torch

def preprocess_yolov8(image_path):
    """
    Load and preprocess an image for YOLOv8/RT-DETR inference.
    Returns the same path (YOLOv8 handles preprocessing internally).
    """
    return image_path


def load_image(image):
    """Convert PIL Image to NumPy array."""
    return np.array(image)


def preprocess_torch(image, device, img_size=(640, 640)):
    """
    Convert PIL image to a normalized Torch tensor.
    Falls back to Pillow resize if OpenCV is not available.
    """
    image = np.array(image)

    if OPENCV_AVAILABLE:
        try:
            image = cv2.resize(image, img_size)
            image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB
        except Exception as e:
            print(f"⚠️ OpenCV resize failed: {e}, falling back to Pillow.")
            image = Image.fromarray(image).resize(img_size)
            image = np.array(image).transpose(2, 0, 1)
    else:
        image = Image.fromarray(image).resize(img_size)
        image = np.array(image).transpose(2, 0, 1)

    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float() / 255.0
    image = image.unsqueeze(0).to(device)
    return image
