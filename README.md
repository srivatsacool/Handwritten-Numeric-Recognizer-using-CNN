# Handdrawn Numeric Recognizer using CNN
#### Made by :- Srivatsa Gorti

---

## Overview

An end-to-end web application for recognizing hand-drawn digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users draw a number directly on an interactive browser canvas, and the model predicts the digit in real time — complete with audio feedback via text-to-speech.

This project is the **digit-focused predecessor** to the full AlphaNumeric Recognizer, demonstrating a clean, production-ready pipeline from CNN inference to a live Streamlit web app. It serves as a strong baseline showing how a classic deep learning model can be transformed into a user-facing product that anyone can interact with — no setup, no code, just draw.

---

## Demo

🔗 **Try it live:** https://srivatsacool-handwritten-numeric-recognizer-using--final-dgcyqm.streamlit.app/

<p align="center">
  <img src="https://user-images.githubusercontent.com/76219802/212967124-f6e2954a-d3dc-4218-bc19-64e925d85630.png" />
</p>

---

## What It Does

- Accepts hand-drawn digit input through a **live browser canvas**
- Classifies digits **0–9** using a CNN trained on MNIST
- Returns predictions **instantly** with audio feedback via gTTS
- Runs entirely in the browser — no installation needed for end users

---

## Why This Project Matters

MNIST digit recognition is one of the most well-known problems in deep learning — but most implementations stop at a Jupyter notebook with a static accuracy score. This project goes further by deploying the model as a **real interactive product**, where users draw their own digits and see the model respond in real time.

It demonstrates a core skill for any ML engineer: bridging the gap between a trained model and a working application that non-technical users can actually use and understand.

---

## How It Works

```
User draws digit on canvas
        ↓
Canvas image captured & preprocessed (28×28 grayscale)
        ↓
CNN model inference (MNIST-trained)
        ↓
Predicted digit returned
        ↓
Result displayed + audio feedback played
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web App Framework | Streamlit |
| Drawing Interface | streamlit-drawable-canvas |
| Deep Learning | TensorFlow / Keras (CNN) |
| Dataset | MNIST (Handwritten Digits) |
| Data Handling | NumPy, Pandas |
| Image Processing | Pillow |
| Audio Feedback | gTTS (Google Text-to-Speech) |

---

## Model Architecture

The CNN is trained on the **MNIST dataset** — 60,000 training images of handwritten digits at 28×28 pixels. The architecture follows a proven convolutional pipeline:

- **Conv2D layers** — extract local spatial features from the input image
- **MaxPooling layers** — downsample and reduce spatial resolution
- **Dropout layers** — regularize to avoid overfitting
- **Flatten + Dense layers** — map learned features to digit classes
- **Softmax output** — probability distribution over 10 digit classes (0–9)

---

## Features

- ✅ Light/dark mode toggle
- ✅ Live canvas drawing with instant prediction
- ✅ Recognizes digits **0–9**
- ✅ Audio feedback on each prediction
- ✅ Cross-platform — runs entirely in the browser

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/srivatsacool/Handwritten-Numeric-Recognizer-using-CNN
cd Handwritten-Numeric-Recognizer-using-CNN

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run final.py
```

### Dependencies

```
streamlit
streamlit_drawable_canvas
numpy
Pillow
tensorflow
pandas
gtts
gtts-token
```

---

## Relationship to AlphaNumeric Recognizer

This project laid the foundation for the more advanced [Handdrawn AlphaNumeric Recognizer](https://github.com/srivatsacool/Handwritten_AlphaNumeric_Recognizer_using_CNN), which extends classification to the full A–Z + 0–9 character set using the EMNIST dataset. Starting with digits-only allowed focused validation of the full app pipeline before scaling up.

---

## Author

**Srivatsa Gorti**
[GitHub](https://github.com/srivatsacool)
