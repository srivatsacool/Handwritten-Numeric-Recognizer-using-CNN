# Handwritten Numeric Recognizer Using CNN

<p align="center">
  <img src="assets/hero.png" alt="Numeric Recognizer Hero Banner" width="100%" />
</p>

## Overview

An end-to-end deep learning web application for recognizing **hand-drawn numbers (0–9)** in real-time. Users draw digits directly on an interactive canvas, and a **Convolutional Neural Network (CNN)** trained on the MNIST dataset predicts the digit with high accuracy — accompanied by satisfying sound effects.

---

## Live Demo

🔗 **Try it out:** [Streamlit App](https://srivatsacool-handwritten-numeric-recognizer-using-cnn-app-yvjqks.streamlit.app/)

---

## Key Features

- **Draw & Recognize** — Freehand canvas for drawing digits 0–9
- **CNN-powered** — Trained on the MNIST dataset for robust digit recognition
- **Sound Effects** — Audio feedback on successful prediction
- **Real-time inference** — Instant results as you draw
- **Interactive UI** — Clean Streamlit interface with drawable canvas

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core language |
| TensorFlow / Keras | CNN model training and inference |
| OpenCV | Image preprocessing |
| NumPy | Array and matrix operations |
| Streamlit | Web application interface |
| streamlit-drawable-canvas | Interactive drawing component |

---

## How It Works

```text
User Draws a Digit
        ↓
Canvas Image Capture
        ↓
Preprocess (grayscale, resize 28×28)
        ↓
CNN Model Inference
        ↓
Digit Prediction + Sound
```

---

## Installation & Setup

```bash
git clone https://github.com/srivatsacool/Handwritten-Numeric-Recognizer-using-CNN
cd Handwritten-Numeric-Recognizer-using-CNN
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Srivatsa Gorti**

---
