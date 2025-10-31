"""
visualization.py
----------------
Handles parsing predictions, drawing bounding boxes with labels, and displaying results.
Supports YOLOv3 (Faster R-CNN baseline), YOLOv8, and RT-DETR.
"""

import cv2
import numpy as np
import streamlit as st

# ------------------------------------------------
# Load COCO label names
# ------------------------------------------------
def load_coco_labels(label_path="data/coco_labels.txt"):
    try:
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"[DEBUG] Loaded {len(labels)} COCO labels.")
    except Exception as e:
        print(f"[ERROR] Could not load COCO labels: {e}")
        labels = []
    return labels


# ------------------------------------------------
# Parse model predictions into boxes, labels, confs
# ------------------------------------------------
def parse_predictions(model_name, preds, conf_threshold=0.25):
    boxes, labels, confidences = [], [], []

    # Ultralytics models: YOLOv8 + RT-DETR
    if model_name in ("YOLOv8", "RT-DETR"):
        for r in preds:
            for box in r.boxes:
                conf = float(box.conf.item())
                if conf >= conf_threshold:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    labels.append(int(box.cls.item()))
                    confidences.append(conf)
        print(f"[DEBUG] Parsed {len(boxes)} boxes from {model_name} output.")

    # TorchVision detection model (YOLOv3/Faster R-CNN baseline)
    elif isinstance(preds, dict):
        scores = preds.get("scores", [])
        labels_list = preds.get("labels", [])
        bboxes = preds.get("boxes", [])
        for score, label, box in zip(scores, labels_list, bboxes):
            if float(score) >= conf_threshold:
                boxes.append(box.cpu().numpy())
                labels.append(int(label))
                confidences.append(float(score))
        print(f"[DEBUG] Parsed {len(boxes)} boxes from TorchVision model.")

    else:
        print(f"[WARN] Unknown prediction type for {model_name}.")

    return boxes, labels, confidences


# ------------------------------------------------
# Draw bounding boxes with labels
# ------------------------------------------------
def draw_boxes(image_np, boxes, labels, confidences, label_map):
    img = image_np.copy()

    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0)
        label_text = (
            f"{label_map[label] if label < len(label_map) else 'Unknown'} {conf:.2f}"
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label_text,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return img


# ------------------------------------------------
# Show results in Streamlit
# ------------------------------------------------
def show_results(image_np, results_dict, conf_threshold=0.25):
    """
    Displays side-by-side detections for all models.
    """
    label_map = load_coco_labels()
    cols = st.columns(len(results_dict))

    for i, (model_name, result) in enumerate(results_dict.items()):
        preds = result.get("preds")
        boxes, labels, confidences = parse_predictions(model_name, preds, conf_threshold)
        results_dict[model_name]["parsed"] = (boxes, labels, confidences)

        annotated_img = draw_boxes(image_np, boxes, labels, confidences, label_map)
        cols[i].image(
            annotated_img,
            caption=f"{model_name} Results",
            use_container_width=True
        )

        st.markdown(f"**{model_name}** detections: {len(boxes)} objects found.")
        print(f"[DEBUG] {model_name} drew {len(boxes)} boxes.")
