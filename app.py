"""
app.py
------
Streamlit app for multi-model object detection comparison.
Supports YOLOv3, YOLOv8, and RT-DETR with side-by-side visualization.
Includes detailed debug logs for first-time loading.
"""

import torch
import streamlit as st
from PIL import Image
import time
from models.model_loader import load_selected_model, run_inference
from utils.preprocessing import load_image
from utils.visualization import show_results
from utils.metrics import measure_inference_time, display_metrics_table

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # avoid GUI dependencies
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"


# ---------------------------
# Streamlit Configuration
# ---------------------------
st.set_page_config(page_title="Multi-Model Object Detection", layout="wide")
st.title("Multi-Model Object Detection Comparator")
st.markdown("Compare YOLOv3, YOLOv8, and RT-DETR on the same image.")

print("[DEBUG] Streamlit app initialized.")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Model Selection")

models_selected = st.sidebar.multiselect(
    "Select models to compare:",
    ["YOLOv3", "YOLOv8", "RT-DETR"],
    default=["YOLOv3", "YOLOv8"]
)

st.sidebar.header("Image Selection")

# Option 1: Upload your own image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Option 2: Use sample images
sample_dir = "data/sample_images"
if os.path.exists(sample_dir):
    sample_files = os.listdir(sample_dir)
    if sample_files:
        selected_sample = st.sidebar.selectbox("Or choose a sample image:", sample_files)
        sample_path = os.path.join(sample_dir, selected_sample)
    else:
        selected_sample, sample_path = None, None
else:
    st.sidebar.warning("Sample images folder not found.")
    selected_sample, sample_path = None, None

print("[DEBUG] Sidebar initialized. Models selected:", models_selected)

# ---------------------------
# Main Image Display
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    print("[DEBUG] Image uploaded.")
elif selected_sample:
    image = Image.open(sample_path).convert("RGB")
    st.image(image, caption=f"Sample: {selected_sample}", use_container_width=True)
    print(f"[DEBUG] Loaded sample image: {selected_sample}")
else:
    st.info("Upload an image or select a sample to begin.")
    print("[DEBUG] Waiting for image input...")
    st.stop()

# ---------------------------
# Run Detection
# ---------------------------
if st.button("Run Detection"):
    print("[DEBUG] Run Detection button clicked.")

    if not models_selected:
        st.warning("Please select at least one model to compare.")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using {device.upper()} for inference.")
    print(f"[DEBUG] Using device: {device}")

    results = {}

    # Progress bar setup
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, model_name in enumerate(models_selected):
        progress_text.text(f"Loading {model_name} model... (this may take a few minutes on first run)")
        print(f"[DEBUG] Loading model: {model_name}")

        with st.spinner(f"Initializing {model_name}... please wait."):
            start_load = time.time()
            model = load_selected_model(model_name, device)
            end_load = time.time()
            print(f"[DEBUG] {model_name} loaded in {end_load - start_load:.2f} seconds.")

        progress_bar.progress(int((i + 0.3) / len(models_selected) * 100))

        progress_text.text(f"Running inference for {model_name}...")
        print(f"[DEBUG] Starting inference for {model_name}")
        preds, inf_time = measure_inference_time(
            run_inference, model_name, model, image, device
        )
        print(f"[DEBUG] {model_name} inference complete - {inf_time} ms")

        results[model_name] = {"preds": preds, "inference_time": inf_time}
        progress_bar.progress(int((i + 1) / len(models_selected) * 100))

    progress_text.text("Inference complete. Generating visualizations...")
    print("[DEBUG] Inference done for all models. Starting visualization...")

    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader("Detection Results")
    image_np = load_image(image)
    show_results(image_np, results)
    print("[DEBUG] Visualization complete.")

    # ---------------------------
    # Metrics Table
    # ---------------------------
    st.subheader("Model Performance")
    display_metrics_table(results)
    print("[DEBUG] Metrics table displayed.")

    progress_text.text("✅ All tasks complete.")
    progress_bar.empty()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    About: This app compares classical (YOLOv3), modern (YOLOv8),
    and transformer-based (RT-DETR) object detectors on the same image.
    Built with PyTorch, Ultralytics, and Streamlit.
    """
)
print("[DEBUG] App ready. Waiting for user interaction.")
