# 🚗 RoadSafe AI – AI-Based Road Accident Detection System

<p align="center">
  <strong>An Intelligent AI-Powered Road Accident Detection and Monitoring System</strong>
</p>

## 📌 Project Overview

**RoadSafe AI** is an AI-based road accident detection system designed to identify road accidents from video footage using Deep Learning and Computer Vision.

The system analyzes video frames and classifies them into two categories: **Accident** and **Non-Accident**. It aims to support intelligent road safety monitoring and contribute to faster accident identification.

## 🎯 Project Objectives

- Detect road accidents automatically from video footage.
- Classify video frames using a Deep Learning model.
- Process videos frame by frame using OpenCV.
- Identify accident-related activities.
- Explore real-time accident monitoring possibilities.
- Improve road safety through AI-powered technology.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| PyTorch | Deep Learning Framework |
| OpenCV | Video Processing |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| CNN | Accident Classification |
| Apple MPS | GPU Acceleration |

## 🧠 AI Model Architecture

RoadSafe AI uses a custom Convolutional Neural Network (CNN) called **AccidentCNN**.

The model consists of:

- Convolutional Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Dropout Regularization
- Softmax-based Prediction

**Input Image Size:** 128 × 128 pixels

**Classification Categories:**
1. Accident
2. Non-Accident

## 📊 Dataset Information

The model was trained and tested using a prepared road accident image dataset.

| Dataset | Images |
|---|---:|
| Training Images | 3,600 |
| Testing Images | 900 |
| Total Images | 4,500 |

**Classes:**
- Accident
- Non-Accident

## 📈 Model Performance

The model was evaluated using 900 test images.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 95.67% |
| Precision | 95.88% |
| Recall | 94.33% |
| F1 Score | 95.04% |

### Confusion Matrix

| Actual / Predicted | Accident | Non-Accident |
|---|---:|---:|
| Accident | 271 | 29 |
| Non-Accident | 10 | 590 |

## 🎥 Video Processing

RoadSafe AI uses OpenCV to process video files frame by frame.

The video processing pipeline includes:

1. Video Loading
2. Frame Extraction
3. Frame Preprocessing
4. AI-Based Accident Detection
5. Prediction and Classification
6. Saving Processed Results

**Successfully processed:** 1,078 video frames during an earlier processing test.

## 📂 Project Structure

```text
RoadSafeAI/
│
├── Data/
│   ├── accident detection/
│   │   ├── prepared_dataset/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── confusion_matrix.png
│   │
│   ├── raw_videos/
│   └── frames/
│
├── scripts/
│   ├── load_test.py
│   ├── extract_frame.py
│   ├── detection.py
│   └── evaluation.py
│
├── models/
│
├── README.md
└── requirements.txt
```

*Note: Adjust the folder and script names above to match your final repository structure.*

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd RoadSafeAI
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the project**

```bash
python3 scripts/load_test.py
```

## 🔬 Future Improvements

- Real-time accident detection using live CCTV footage.
- Integration with GPS-based location tracking.
- Automatic emergency alert notifications.
- Improved accident detection accuracy.
- Web-based monitoring dashboard.
- Integration with emergency response services.

## 👨‍💻 Developer

**Utsho Kumar Deb**

Computer Science & Engineering (CSE)  
Northern University Bangladesh

**Project:** RoadSafe AI  
**Domain:** Artificial Intelligence | Deep Learning | Computer Vision

---

<p align="center">
  <strong>RoadSafe AI – Making Roads Safer Through Artificial Intelligence.</strong>
</p>
