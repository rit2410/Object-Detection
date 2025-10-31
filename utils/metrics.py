"""
metrics.py
------------
Utility functions to compute and display model performance metrics:
- Inference time (ms)
- Frames per second (FPS)
- Optional: mAP evaluation for object detection (if ground truth is available)
"""

import time
import torch
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# 1. Timing Utilities
# ---------------------------
def measure_inference_time(model_func, *args, **kwargs):
    """
    Measures inference time (in milliseconds) for a given model function.
    Returns:
        (result, elapsed_time_ms)
    """
    start = time.perf_counter()
    result = model_func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    return result, round(elapsed_ms, 2)


def compute_fps(inference_time_ms):
    """
    Compute frames per second (FPS) from inference time (ms per image).
    """
    if inference_time_ms <= 0:
        return 0.0
    return round(1000 / inference_time_ms, 2)


# ---------------------------
# 2. Optional: mAP Evaluation
# ---------------------------
def compute_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Simplified mAP@IoU calculation for small-scale validation sets.
    Args:
        pred_boxes: list of [x1, y1, x2, y2, conf, class_id]
        true_boxes: list of [x1, y1, x2, y2, class_id]
        iou_threshold: IoU threshold for true positive
    Returns:
        float: mean Average Precision
    """
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0.0

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)  # sort by confidence
    tp, fp = 0, 0
    detected = []

    for pbox in pred_boxes:
        best_iou = 0
        best_gt = None
        for i, gt in enumerate(true_boxes):
            if gt[4] == pbox[5]:
                iou = bbox_iou(pbox[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
        if best_iou >= iou_threshold and best_gt not in detected:
            tp += 1
            detected.append(best_gt)
        else:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (len(true_boxes) + 1e-6)
    return round((precision * recall) / (precision + recall + 1e-6) * 2, 4)  # F1 approximation


def bbox_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# ---------------------------
# 3. Streamlit Table Display
# ---------------------------
def display_metrics_table(results_dict):
    """
    Displays metrics in a neat Streamlit dataframe.
    Args:
        results_dict: dict of model_name → {"inference_time": ms, "map": val, ...}
    """
    table = {
        "Model": [],
        "Inference (ms)": [],
        "FPS": [],
        "mAP@50": []
    }

    for model_name, res in results_dict.items():
        inf_time = res.get("inference_time", 0)
        map_score = res.get("map", "-")
        table["Model"].append(model_name)
        table["Inference (ms)"].append(inf_time)
        table["FPS"].append(compute_fps(inf_time))
        table["mAP@50"].append(map_score)

    df = pd.DataFrame(table)
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(df, use_container_width=True)
