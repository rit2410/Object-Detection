"""
Preprocessing utilities for object detection app.
Handles image loading, resizing, normalization, and tensor conversion
to ensure compatibility across YOLOv3, YOLOv8, and RT-DETR models.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2

# ---------------------------
# 1. Load image (OpenCV / PIL)
# ---------------------------
def load_image(image_source):
    """
    Loads an image from file path or PIL object and converts to RGB numpy array.
    """
    if isinstance(image_source, Image.Image):
        image = np.array(image_source.convert("RGB"))
    else:
        image = cv2.imread(image_source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# ---------------------------
# 2. Preprocess for YOLOv3 / RT-DETR (PyTorch models)
# ---------------------------
def preprocess_torch(image, device, size=640):
    """
    Prepares an image for PyTorch-based detectors (YOLOv3, RT-DETR).
    - Resize to square
    - Normalize to [0,1]
    - Convert to CHW tensor
    """
    # Resize
    resized = cv2.resize(image, (size, size))
    # Normalize
    normalized = resized / 255.0
    # Convert HWC → CHW
    chw = np.transpose(normalized, (2, 0, 1))
    # Convert to tensor
    tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor


# ---------------------------
# 3. Preprocess for YOLOv8 (Ultralytics handles it internally)
# ---------------------------
def preprocess_yolov8(image_path_or_pil):
    """
    YOLOv8 handles preprocessing internally during .predict().
    This function just ensures we provide a valid input path or PIL image.
    """
    if isinstance(image_path_or_pil, str):
        return image_path_or_pil  # image path
    elif isinstance(image_path_or_pil, Image.Image):
        return image_path_or_pil  # PIL image is accepted
    else:
        raise ValueError("Invalid input type for YOLOv8 preprocessing.")


# ---------------------------
# 4. Optional: Video frame extraction (for future use)
# ---------------------------
def extract_frames(video_path, skip_frames=5):
    """
    Generator function to read frames from a video file.
    Yields every nth frame (based on skip_frames) in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1
    cap.release()
