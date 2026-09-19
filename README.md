# Signova

### Real Time Sign Language Recognition & Speech Assistance

- Signova is a computer vision based sign language recognition system designed to recognize hand signs in real time and convert them into meaningful outputs.

- The project combines **MediaPipe hand tracking**, **deep learning**, **OpenCV**, and **speech synthesis** to create an interactive sign-to-speech pipeline.

---

## Overview

- Communication barriers can make everyday interactions difficult for people who rely on sign language.

- Signova explores a technology driven approach to bridging this gap by using a camera to detect hand gestures, identify the corresponding sign, and provide an accessible output.

The system follows the pipeline:

**Camera → Hand Detection → Feature Extraction → Sign Classification → Prediction → Speech Output**

---

## Features

- Real-time hand gesture detection
- Computer vision-based sign recognition
- Deep learning classification model
- Support for alphabetic and numeric signs
- Live camera input
- Trained `.keras` model
- Speech output integration
- Dataset collection and training utilities
- Modular Python architecture

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| TensorFlow / Keras | Deep learning model |
| MediaPipe | Hand landmark detection |
| OpenCV | Camera and image processing |
| NumPy | Numerical computation |
| Pygame | Audio interaction |
| Sarvam AI | Speech / language integration |

---

## Project Structure

```text
Signova/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── ...
│   ├── X/
│   ├── Y/
│   └── Z/
│
├── model/
│   └── sign_language.keras
│
├── src/
│   ├── app.py
│   ├── camera.py
│   ├── collect.py
│   ├── gui.py
│   ├── predict.py
│   ├── predictor.py
│   ├── sarvam_tts.py
│   ├── test_sarvam.py
│   ├── test_voice.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
