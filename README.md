# Deepfake-Detection

ELA-Based Image Manipulation Detection Using CNN

This repository contains a Flask-based web application that performs image manipulation detection using Error Level Analysis (ELA) and a Convolutional Neural Network (CNN) model. The system processes uploaded images, converts them into ELA representations, and classifies them as either real or fake based on a pre-trained CNN model.

The core methodology leverages ELA to highlight compression artifacts typically introduced by tampering operations. These ELA images are then fed into a deep learning model trained on tampered and authentic image datasets. The web interface is responsive, supports drag-and-drop functionality, and enables real-time predictions.

This project is part of an academic research initiative aimed at improving digital image forensics through machine learning techniques. It demonstrates the integration of traditional image forensic methods (ELA) with modern deep learning approaches for practical deployment.

Key Features:

ELA preprocessing for JPEG images

Lightweight CNN architecture with softmax classification

Interactive and responsive web UI (Bootstrap-based)

Support for drag-and-drop and multi-image input

Deployable on Railway, Render, or other cloud platforms

Frameworks & Tools:
Flask, TensorFlow, Pillow, NumPy, Bootstrap 5
