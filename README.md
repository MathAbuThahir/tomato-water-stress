# tomato-water-stress
# 🍅 Tomato Leaf Water Stress Detection using YOLOv8 ONNX (Raspberry Pi)

This project implements a **real-time tomato leaf water stress detection system** using a **YOLOv8 object detection model (ONNX)** deployed on a **Raspberry Pi 4B** with the **Camera Module 3**. The system detects water stress symptoms in tomato leaves and displays live predictions through a local web interface built using Flask.

---

## 🧠 Project Summary

- **Objective**: Detect and monitor water stress in tomato plants to aid in precision agriculture.
- **Model**: YOLOv8 trained on annotated tomato leaf images with a single class: `"Stress"`.
- **Hardware**: Raspberry Pi 4B, Camera Module 3 (mounted on a pole for top view), running detection in real-time.
- **Deployment**: Lightweight ONNX inference with `onnxruntime` and streaming via a Flask web server.
- **Live Preview**: Detects "Stress" on leaves and updates a live video feed with colored indicators:
  - 🟡 No leaf detected
  - 🟢 Leaf detected, no stress
  - 🔴 Stress detected

---

## 🔧 Hardware & Software Requirements

### Hardware:
- Raspberry Pi 4B (4GB or 8GB)
- Raspberry Pi Camera Module 3
- MicroSD card (32GB+)
- Power supply or USB-C power bank
- Mount (e.g., pole or stand) for fixed camera positioning

### Software:
- Python 3.9+
- `onnxruntime`
- `opencv-python`
- `flask`
- `picamera2` (libcamera-based capture)

Install dependencies:
```bash
pip install -r requirements.txt
