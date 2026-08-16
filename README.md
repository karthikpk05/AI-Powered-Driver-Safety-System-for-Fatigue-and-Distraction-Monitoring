# 🚗 AI-Powered-Driver-Safety-System-for-Fatigue-and-Distraction-Monitoring

### Fatigue and Distraction Monitoring Using Computer Vision & Deep Learning

An AI-powered real-time **Driver Safety System (DSS)** designed to detect driver fatigue and distraction using computer vision and deep learning. The system monitors facial features and driver behavior through a camera and provides timely alerts when unsafe driving conditions are detected.

The project focuses on detecting:

* 👁️ **Eye closure / prolonged blinking**
* 🥱 **Yawning**
* 📱 **Mobile phone usage**
* 🚨 **Overall driver alertness state**

The system classifies the driver's condition into **Awake, Tired, Drowsy, and Critical** states and generates visual or voice alerts when unsafe behavior is detected.

---

## 📌 Project Overview

Driver fatigue and distraction are important contributors to road accidents. Long driving hours can reduce alertness, while activities such as mobile phone usage can divert the driver's attention from the road.

This project proposes a **non-intrusive, vision-based driver monitoring system** that continuously analyzes live video from a camera. Facial landmarks and deep learning models are used to identify visual indicators of fatigue and distraction.

The system combines:

* Computer Vision
* Deep Learning
* Facial Landmark Detection
* CNN-based Classification
* Object Detection
* Temporal Decision Logic
* Real-Time Alerting

The overall goal is to provide a supportive safety mechanism that can warn drivers before fatigue or distraction leads to dangerous situations.

---

## ✨ Key Features

### 👁️ Eye State Detection

A CNN model analyzes the driver's eye region and classifies the eyes as:

* Open
* Closed

Prolonged eye closure is treated as an indicator of fatigue. The eye-state model was evaluated on **1,528 images** and achieved an accuracy of **99.61%**.

### 🥱 Yawning Detection

A second CNN model analyzes the driver's mouth region to detect yawning behavior.

The yawning dataset contains **604 mouth images**, categorized into yawning and non-yawning classes. The model achieved an accuracy of **99.34%** during evaluation.

### 📱 Mobile Phone Detection

A pretrained **YOLOv8 object detection model** is used to identify mobile phones in the camera frame.

When a phone is detected near the driver or in the driver's hand, the system considers this a distraction and generates a warning.

### 🚨 Driver State Classification

The system combines eye closure, yawning, and distraction information with temporal analysis to determine the driver's condition.

The driver can be classified into:

| State           | Description                                    |
| --------------- | ---------------------------------------------- |
| 🟢 **Awake**    | Driver shows no significant fatigue indicators |
| 🟡 **Tired**    | Initial signs of fatigue detected              |
| 🟠 **Drowsy**   | Multiple fatigue indicators detected           |
| 🔴 **Critical** | Severe/repeated fatigue indicators detected    |

The report's fatigue classification logic uses long blinks and yawning frequency to determine the fatigue level.

### 🔊 Real-Time Alerts

When unsafe behavior is detected, the system can provide:

* Voice notifications
* Visual warning messages
* Driver status information
* Real-time monitoring dashboard

The dashboard can display information such as blink count, yawning frequency, and current driver status.

---

## 🏗️ System Architecture

The system follows a layered architecture:

```text
                    ┌──────────────────────┐
                    │      Camera Input     │
                    │   Live Video Stream   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Processing Layer   │
                    │ Face Detection        │
                    │ MediaPipe Face Mesh   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │ Eye CNN    │ │ Yawn CNN   │ │  YOLOv8    │
          │ Open/Closed│ │ Yawn/Normal │ │ Phone      │
          └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Decision / Temporal │
                    │       Analysis       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Driver State       │
                    │ Awake / Tired /      │
                    │ Drowsy / Critical    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Alerts & Dashboard │
                    │ Visual + Voice Alert │
                    └──────────────────────┘
```

The architecture consists of input, processing, AI detection, decision, and output layers.

---

## 🛠️ Technology Stack

| Technology              | Purpose                         |
| ----------------------- | ------------------------------- |
| **Python**              | Core development                |
| **OpenCV**              | Real-time video processing      |
| **TensorFlow**          | Deep learning model development |
| **CNN**                 | Eye and yawning classification  |
| **MediaPipe Face Mesh** | Facial landmark detection       |
| **YOLOv8**              | Mobile phone object detection   |
| **Webcam**              | Real-time video input           |

These technologies are explicitly used in the implementation described in the project report.

---

## 🔄 How It Works

```text
1. Camera captures live video
              ↓
2. Driver face is detected
              ↓
3. Facial landmarks are extracted
              ↓
4. Eye and mouth regions are cropped
              ↓
5. Eye CNN detects open/closed state
              ↓
6. Yawning CNN detects yawning
              ↓
7. YOLOv8 detects mobile phone usage
              ↓
8. Temporal logic analyzes the detected events
              ↓
9. Driver state is classified
              ↓
10. Warning/alert is generated if required
```

The system continuously processes video frames so that fatigue and distraction can be detected during operation.

---

## 📊 Model Performance

### Eye State Classification

| Metric       |        Result |
| ------------ | ------------: |
| Dataset Size |  1,528 images |
| Classes      | Open / Closed |
| Accuracy     |    **99.61%** |
| Model        |           CNN |

### Yawning Detection

| Metric       |                Result |
| ------------ | --------------------: |
| Dataset Size |            604 images |
| Classes      | Yawning / Non-Yawning |
| Accuracy     |            **99.34%** |
| Model        |                   CNN |

The report also states that precision, recall, and F1-score were close to 1.00 for eye classification and approximately 0.99 for yawning classification.

> **Note:** These figures represent the evaluation results reported in the project paper and should not be interpreted as guaranteed real-world driving accuracy.

---


## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Powered-Driver-Safety-System.git
cd AI-Powered-Driver-Safety-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Connect a webcam and run the main application:

```bash
python src/main.py
```

The system will capture live video and continuously monitor the driver's facial features and behavior.

> The exact command may need to be modified according to the entry-point file in the repository.

---

## 🧠 Fatigue Detection Logic

The system combines multiple visual indicators rather than relying on a single feature.

```text
Eye Closure
     +
Yawning Frequency
     +
Temporal Analysis
     ↓
Driver Fatigue Level
```

Mobile phone detection is separately incorporated as a distraction indicator.

This multi-indicator approach allows the system to continuously monitor both fatigue and distraction.

---

## 📸 System Output

The project demonstrates:

* Real-time eye-state detection
* Yawning detection
* Mobile phone detection with bounding boxes
* Driver-state classification
* Warning messages
* Real-time monitoring information

The report includes demonstrations of eye detection, yawning detection, mobile-phone detection, and driver-state classification.

---

## 🚀 Future Improvements

Possible future improvements identified in the project include:

* Deployment on embedded platforms
* Integration directly into vehicles
* Detection of additional risky behaviors
* Smoking detection
* Inattentive head-movement detection
* Further improvements to detection accuracy and usability

These extensions could make the system more comprehensive for real-world driver monitoring applications.

---

## 🎓 Academic Project

This project was developed as a final-year academic project at:

**Saranathan College of Engineering**
Department of Computer Science and Business Systems
Tiruchirappalli, India


## ⚠️ Disclaimer

This project is intended as an **academic and research prototype** for driver fatigue and distraction monitoring.


---

## 📄 Project Report

The complete project report is included in this repository:

```text
project-report.pdf
```

---

## ⭐ Acknowledgement

We would like to express our sincere gratitude to the Department of Computer Science and Business Systems, Saranathan College of Engineering, and our project guide for their guidance, encouragement, and valuable suggestions throughout the development of this project.

---
