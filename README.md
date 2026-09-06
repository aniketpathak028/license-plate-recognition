# Real-Time License Plate Recognition (LPR) System

detect, read, and stabilize car license plates in real time using  videos

## Dataset & Model

* **Model:** [YOLOv8n (YOLOv8 nano)](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) used for core object detection.
* **Dataset:** [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) utilized for training and fine-tuning the model weights.

## How It Works

* **Frame Capture:** OpenCV reads an input video file (`.mp4` or `.mov`) frame by frame from start to finish.
* **Plate Detection:** A fine-tuned YOLOv8 nano model scans each frame to locate any car license plates and draws a bounding box around them.
* **Image Preprocessing:** OpenCV crops out the detected plate area, converts it to grayscale, applies Otsu's thresholding to turn it into high-contrast black and white, and doubles its size for better clarity.
* **Text Reading (OCR):** EasyOCR scans the cleaned image to extract alphanumeric text, using a strict allowlist to block unwanted symbols.
* **Format Correction & Stabilization:** A custom logic script fixes common OCR confusions (like confusing the letter 'O' with the number '0'). A rolling history queue then uses **majority voting** across recent frames to keep the text rock-solid and prevent flickering.
* **Video Annotation:** The script draws a green outline around the plate, attaches a zoomed-in picture-in-picture preview above the car, prints the clean text label, and compiles everything into a brand-new playable video.

## Tech Stack

* **Object Detection:** YOLOv8 (Ultralytics) fine-tuned on custom plate data
* **Text Extraction:** EasyOCR
* **Video & Image Processing:** OpenCV (`cv2`)
* **Environment:** Google Colab with Google Drive storage

## How to Run the Project

1. click the link to visit the webapp hosted on streamlit
2. upload a video containing cars with number plates and click on process!

## Demo
