# 🧠 Real-Time Edge Vision System for Object Detection and Depth Estimation

A lightweight **computer vision pipeline** designed for **Raspberry Pi 4**, performing real-time **object detection** and **depth estimation** using **YOLOv8** and **FastDepthV2**.  
This project demonstrates how optimized models and ROI-based logic can bring AI-powered perception to **low-power edge devices**.

---

## 🚀 Features

- 🔍 **Dual Inference:** Real-time RGB object detection + monocular depth estimation  
- ⚡ **Optimized for Edge:** Achieves ~8–10 FPS on Raspberry Pi 4 using YOLOv8n and FastDepthV2  
- 🎯 **ROI-Based Depth Trigger:** Reduces false detections by ~20–25% using selective depth validation  
- 🧵 **Multi-Threaded Pipeline:** Asynchronous frame capture and inference for low-latency response  
- 📊 **Performance Benchmarks:** Latency, accuracy, and power usage tested across ONNX optimization levels  
- 🪟 **Embedded Monitoring UI:** Real-time frame visualization with detection overlays, FPS tracking, and class confidence display  

---

## 🧰 Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Language** | Python |
| **Core Libraries** | OpenCV, NumPy, PyTorch, Ultralytics YOLO |
| **Depth Estimation** | FastDepthV2 |
| **Model Optimization** | ONNX Runtime |
| **Hardware** | Raspberry Pi 4, Waveshare IMX335 USB Camera |

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/edge-vision-detection.git
cd edge-vision-detection
