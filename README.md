# 🌸 Flower Classifier – Deep Learning with Docker

This project is an image classification system that predicts the type of a flower from an uploaded image.
The model is trained using TensorFlow + Keras and packaged with Docker.

---

## 🧠 Model Overview

- Base Model: EfficientNetB0 (pretrained on ImageNet)
- Framework: TensorFlow / Keras
- Input Size: 224 x 224 RGB images
- Output: Flower class + confidence score
- Training Method:
  - Transfer Learning
  - Fine-tuning enabled
  - Data augmentation applied

---

## 🌼 Supported Classes

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

---

## 📁 Project Structure

```text
flower-classifier-docker/
├── app/
│ ├── model/
│ │ └── flower_model.keras
│ ├── static/
│ │ └── index.html
│ ├── main.py
│ └── requirements.txt
├── Dockerfile
├── KERAS_LICENSE.txt
├── SCIKIT-LEARN_LICENSE.txt
└── README.md
```

---

## 🐳 Docker Usage

### 🔨 Build Docker Image

```text
cd ~/flower-classifier-docker
docker build -t flower-classifier .
```

---

### ▶️ Run Docker Container (CPU)

```text
docker run -p 8000:8000 flower-classifier
```

---

### 🚀 Run with GPU Support (Optional)

```text
docker run --gpus=all -p 8000:8000 bugramurat/flower-classifier-app:1.0
```

Make sure NVIDIA Container Toolkit is installed.

---

## 🧪 Model Training Summary

- Optimizer: Adam
- Learning Rate: 1e-5
- Loss Function: Categorical Crossentropy
- Callbacks:
  - EarlyStopping
  - ModelCheckpoint

The model is saved using the native Keras `.keras` format.

---

## 🖼️ Image Preprocessing

Each image is:

1. Converted to RGB
2. Resized to 224x224
3. Normalized to [0, 1]
4. Expanded to batch dimension

---

## ⚠️ Notes

- CUDA warnings inside Docker are normal if GPU is not available
- Port 8000 must be free before running
- If the app still runs after stopping Docker, another process may be using the port

---

## 👤 Author

Bugra MURAT

Deep Learning · Computer Vision · Docker
