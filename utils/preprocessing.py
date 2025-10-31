import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Try importing OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception as e:
    print(f"⚠️ OpenCV failed to load: {e}")
    cv2 = None
    OPENCV_AVAILABLE = False

import numpy as np
from PIL import Image
import torch


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
