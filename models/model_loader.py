"""
model_loader.py
----------------
Centralized interface to load and run inference on multiple detection models.
Supported models: YOLOv3, YOLOv8, RT-DETR
"""

import torch
from models.yolo_v3 import load_model as load_yolo_v3, detect as detect_yolo_v3
from models.yolo_v8 import load_model as load_yolo_v8, detect as detect_yolo_v8
from models.rt_detr import load_model as load_rt_detr, detect as detect_rt_detr


import streamlit as st
from models.yolo_v3 import load_model as load_yolo_v3, detect as detect_yolo_v3
from models.yolo_v8 import load_model as load_yolo_v8, detect as detect_yolo_v8
from models.rt_detr import load_model as load_rt_detr, detect as detect_rt_detr

@st.cache_resource
def load_selected_model(model_name, device="cpu"):
    """Load and cache model by name."""
    if model_name == "YOLOv3":
        model, _ = load_yolo_v3(device)
        return model
    elif model_name == "YOLOv8":
        return load_yolo_v8("yolov8n.pt")
    elif model_name == "RT-DETR":
        return load_rt_detr(model_variant="rtdetr-l.pt")
    else:
        raise ValueError(f"Unsupported model: {model_name}")



# ---------------------------
# 2. Run Inference
# ---------------------------
def run_inference(model_name, model, image, device="cpu"):
    """
    Unified inference function for any supported model.
    Args:
        model_name (str): Model identifier
        model: Loaded model object
        image: PIL or path
        device: "cpu" or "cuda"
    Returns:
        predictions (dict or Results): Model-specific predictions
    """
    if model_name == "YOLOv3":
        preds = detect_yolo_v3(model, image, device)

    elif model_name == "YOLOv8":
        preds = detect_yolo_v8(model, image)

    elif model_name == "RT-DETR":
        preds = detect_rt_detr(model, image)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return preds


# ---------------------------
# 3. Optional Utility: Model Summary
# ---------------------------
def get_model_info(model_name, model):
    """
    Prints or returns basic model info (params, type, etc.)
    """
    info = {}
    if hasattr(model, "num_parameters"):
        info["parameters"] = model.num_parameters()
    else:
        info["parameters"] = sum(p.numel() for p in model.parameters()) / 1e6

    info["architecture"] = model_name
    return info
