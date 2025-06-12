# 🧠 ELA-Based Image Manipulation Detection Using CNN

This repository contains a Flask-based web application that performs image manipulation detection using **Error Level Analysis (ELA)** and a **Convolutional Neural Network (CNN)** model. It enables users to upload images through a user-friendly web interface and classifies each image as either **real** or **fake** based on forensic compression artifacts.

🌐 **Live Demo**: [https://deepfake-detect.up.railway.app/](https://deepfake-detect.up.railway.app/)

## 📚 Overview

The system applies ELA to JPEG images to visualize potential regions of tampering. These enhanced images are then passed through a lightweight CNN architecture trained on labeled datasets to perform binary classification.

This project serves as an academic implementation to explore the effectiveness of integrating traditional image forensics with deep learning for detecting image manipulation.

---

## ✨ Features

- 🔍 ELA preprocessing to enhance tampering clues in images  
- 🤖 CNN-based classification for real vs. fake detection  
- 🌐 Web UI with Bootstrap
- 🧪 Softmax output visualization (internally logged)
- ☁️ Ready for deployment on platforms like Railway or Render

---

## 🛠️ Tech Stack

- **Backend:** Flask, TensorFlow, NumPy, Pillow  
- **Frontend:** HTML, CSS, Bootstrap 5, JavaScript  
- **Model:** Sequential CNN with ELA-preprocessed image inputs  
- **Deployment:** Railway

---

## 🧪 How It Works

1. **Upload an Image:** JPEG image is submitted via the form or drag-and-drop zone.
2. **ELA Conversion:** Image is saved temporarily and recompressed; the ELA image is generated via pixel-wise difference.
3. **Prediction:** The ELA image is resized, normalized, and passed to the CNN model.
4. **Output:** The model outputs a softmax prediction displayed on the webpage.

---

## 📁 Project Structure
├── app.py # Main Flask application

├── model.py # CNN architecture and ELA function

├── templates/

│ └── index.html # Web UI (Bootstrap-based)

├── static/ # Uploaded images

├── model/

│ └── best_model_converted.h5 # Trained CNN model

└── requirements.txt # Python dependencies

---

This project is intended for academic and research purposes only. 
