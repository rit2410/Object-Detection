# Multi-Model Object Detection App (YOLOv3 · YOLOv8 · RT-DETR)

A **Streamlit web app** that lets you compare the performance of three powerful object detection models side-by-side:

- **YOLOv3** — Classic CNN-based detector  
- **YOLOv8** — Modern, real-time object detector  
- **RT-DETR** — Transformer-based, end-to-end detector

---

## Features

1. Upload your own images or use sample images
2. Run inference with YOLOv3, YOLOv8, or RT-DETR
3. See detections side-by-side with bounding boxes & labels
4. Compare **inference time**, **confidence**, and **accuracy**
5. Fully interactive UI built with **Streamlit**

---

## Folder Structure

```
📦 CNN-ObjectDetection-App
┣ 📂 models
┃ ┣ yolo_v3.py
┃ ┣ yolo_v8.py
┃ ┣ rt_detr.py
┃ ┗ model_loader.py
┣ 📂 utils
┃ ┣ preprocessing.py
┃ ┣ visualization.py
┃ ┗ metrics.py
┣ 📂 data
┃ ┗ 📂 sample_images
┃ ┗ example.jpg
┣ 📜 app.py
┣ 📜 coco_labels.txt
┣ 📜 requirements.txt
┣ 📜 README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/CNN-ObjectDetection-App.git
cd CNN-ObjectDetection-App

2️⃣ Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # for macOS/Linux
venv\Scripts\activate     # for Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the App
streamlit run app.py
Then open the provided local URL (e.g. http://localhost:8501) in your browser.

| Model   | Type              | Source                                                            |
| ------- | ----------------- | ----------------------------------------------------------------- |
| YOLOv3  | CNN-based         | `torchvision.models.detection`                                    |
| YOLOv8  | Modern Real-Time  | [Ultralytics](https://github.com/ultralytics/ultralytics)         |
| RT-DETR | Transformer-based | [Hugging Face – RT-DETR](https://huggingface.co/lyuwenyu/RT-DETR) |





