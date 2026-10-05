# 🌿 Edge-AI Crop Disease Detector

[![Live Demo](https://img.shields.io/badge/Demo-Live_App-2e7d32?style=for-the-badge&logo=render)](https://crop-disease-detector-r6cb.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)

An optimized, end-to-end Computer Vision application engineered for offline crop disease diagnosis. Powered by lightweight MobileNetV2 architecture quantized to TensorFlow Lite, served via FastAPI, and deployed on Render with interactive agricultural treatment plans.

🚀 **Live Web Application:** [https://crop-disease-detector-r6cb.onrender.com](https://crop-disease-detector-r6cb.onrender.com)

---

## 💡 Key Features

- **Edge-AI Optimization:** Quantized MobileNetV2 CNN model (`.tflite`) designed for low-latency, edge-device processing without heavy GPU requirements.
- **FastAPI Microservice:** High-performance REST backend handling image preprocessing, TFLite inference execution, and dynamic response generation.
- **Interactive UI Dashboard:** Embedded responsive drag-and-drop dashboard with dynamic image preview and real-time accuracy progress metrics.
- **Actionable Agronomic Advice:** Instant return of categorized diagnosis, symptoms, chemical treatments (fungicides), and preventative crop management guidelines.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning & CV:** TensorFlow 2.x, Keras, TensorFlow Lite, PIL (Pillow), NumPy
- **Backend Framework:** FastAPI, Uvicorn (ASGI)
- **Frontend Presentation:** HTML5, CSS3, JavaScript (Fetch API), Bootstrap 5, FontAwesome
- **Deployment & Infra:** Render Cloud Platform, GitHub CI/CD Flow, Cron-job Uptime Keepalive

---

## 📂 Project Structure

```text
crop-disease-detector/
│
├── models/                   # Quantized TFLite edge model
│   └── crop_model.tflite
│
├── app.py                    # FastAPI server, API endpoints & embedded UI
├── train.py                  # ML pipeline, MobileNetV2 transfer learning & TFLite conversion
├── requirements.txt          # Production dependencies
├── render.yaml               # Infrastructure as Code (Render deployment)
├── .python-version           # Specified Python 3.10.12 runtime environment
└── README.md                 # Technical documentation
