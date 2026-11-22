
import os
import re
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import easyocr

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data", "accident detection")

ACCIDENT_DIR = os.path.join(DATA_DIR, "accident")
NON_ACCIDENT_DIR = os.path.join(DATA_DIR, "non_accident")

MODEL_PATH = os.path.join(DATA_DIR, "accident_model.pth")

# General vehicle detection model
YOLO_MODEL_PATH = "yolo11n.pt"

# Separate trained license plate detection model
PLATE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "license_plate_detector.pt"
)

SUPPORTED_VIDEO = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASS_NAMES = ["Accident", "Non-Accident"]

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RoadSafe AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 3. DASHBOARD THEME
# =====================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #0B1120;
    color: #E2E8F0;
}

.block-container {
    max-width: 1650px;
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #26364D;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.main-header {
    background: linear-gradient(
        120deg, #101827, #142B38, #153D46
    );
    border: 1px solid #28505A;
    padding: 28px;
    border-radius: 17px;
    margin-bottom: 22px;
}

.main-header h1 {
    color: #F8FAFC;
    font-size: 32px;
    font-weight: 800;
}

.main-header p {
    color: #A5C3C5;
}

div[data-testid="stMetric"] {
    background: #172337;
    border: 1px solid #293B53;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
    font-weight: 800;
}

div.stButton > button {
    background: #153D46;
    color: white;
    border: 1px solid #287D7A;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

div.stButton > button:hover {
    background: #1B5960;
    border-color: #2DD4BF;
    color: white;
}

.accident-status {
    background: #351923;
    border: 1px solid #EF4444;
    color: #FCA5A5;
    padding: 13px;
    border-radius: 10px;
    font-weight: 700;
}

.safe-status {
    background: #123528;
    border: 1px solid #22C55E;
    color: #86EFAC;
    padding: 13px;
    border-radius: 10px;
    font-weight: 700;
}

hr {
    border-color: #293B53;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. CNN ACCIDENT MODEL
# =====================================================

class AccidentCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        return self.classifier(self.features(x))


# =====================================================
# 5. LOAD MODELS
# =====================================================

@st.cache_resource
def load_cnn_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    model = AccidentCNN().to(DEVICE)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.eval()

    return model


@st.cache_resource
def load_yolo_model():

    return YOLO(YOLO_MODEL_PATH)


@st.cache_resource
def load_plate_model():

    if not os.path.exists(PLATE_MODEL_PATH):
        return None

    return YOLO(PLATE_MODEL_PATH)


@st.cache_resource
def load_ocr_reader():

    return easyocr.Reader(
        ["en"],
        gpu=(DEVICE == "cuda")
    )


# =====================================================
# 6. VIDEO UTILITIES
# =====================================================

def get_video_files(folder):

    if not os.path.isdir(folder):
        return []

    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(SUPPORTED_VIDEO)
    ]


def video_name(path):

    return os.path.basename(path) if path else "No video selected"


def adjust_brightness(frame, brightness):

    return cv2.convertScaleAbs(
        frame,
        alpha=brightness,
        beta=0
    )


def zoom_frame(frame, zoom):

    if frame is None or zoom <= 1:
        return frame

    height, width = frame.shape[:2]

    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    cropped = frame[
        y1:y1 + crop_height,
        x1:x1 + crop_width
    ]

    return cv2.resize(cropped, (width, height))


# =====================================================
# 7. CNN PREDICTION
# =====================================================

def predict_accident(model, frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    tensor = IMAGE_TRANSFORM(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output, dim=1
        )

        confidence, prediction = torch.max(
            probabilities, dim=1
        )

    return (
        CLASS_NAMES[prediction.item()],
        float(confidence.item()) * 100
    )


# =====================================================
# 8. YOLO VEHICLE DETECTION
# =====================================================

def detect_vehicles(model, frame):

    results = model.predict(
        frame,
        conf=0.35,
        verbose=False
    )

    annotated = frame.copy()

    count = 0

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            if class_id not in VEHICLE_CLASSES:
                continue

            count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            label = VEHICLE_CLASSES[class_id]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 220, 150),
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 150),
                2
            )

    return annotated, count


# =====================================================
# 9. NUMBER PLATE DETECTION + OCR
# =====================================================

def detect_number_plates(
    plate_model,
    reader,
    frame
):

    annotated = frame.copy()

    detected_plates = []

    if plate_model is None or reader is None:
        return annotated, detected_plates

    results = plate_model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            height, width = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # OCR reads the plate text
            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            ocr_results = reader.readtext(
                gray,
                detail=0,
                paragraph=False
            )

            plate_text = " ".join(
                str(text).strip()
                for text in ocr_results
                if str(text).strip()
            )

            # Clean OCR result
            plate_text = re.sub(
                r"[^A-Za-z0-9 -]",
                "",
                plate_text
            ).strip()

            if not plate_text:
                plate_text = "Text Not Read"

            detected_plates.append(plate_text)

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                3
            )

            cv2.putText(
                annotated,
                plate_text[:35],
                (x1, max(y1 - 12, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 180, 0),
                2
            )

    return annotated, detected_plates


# =====================================================
# 10. SESSION STATE
# =====================================================

DEFAULTS = {

    "history": [],

    "playing": False,

    "cap_accident": None,
    "cap_normal": None,

    "accident_path": None,
    "normal_path": None,

    "cnn_model": None,
    "yolo_model": None,
    "plate_model": None,
    "ocr_reader": None,

    "frame_count": 0,
    "vehicle_count": 0,
    "accident_alerts": 0,

    "accident_done": False,
    "normal_done": False,

    "accident_frame": None,
    "normal_frame": None,

    "accident_prediction": "Waiting",
    "normal_prediction": "Waiting",

    "accident_confidence": 0.0,
    "normal_confidence": 0.0,

    "accident_vehicle_count": 0,
    "normal_vehicle_count": 0,

    "accident_plates": [],
    "normal_plates": [],

    "all_plate_texts": [],

    "session_started": None,

    "report_saved": False,

    "last_error": None,

    "last_alert_frame": -1
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 11. VIDEO CONTROL FUNCTIONS
# =====================================================

def release_videos():

    for key in [
        "cap_accident",
        "cap_normal"
    ]:

        cap = st.session_state.get(key)

        if cap is not None:
            cap.release()

        st.session_state[key] = None


def reset_detection():

    values = {

        "frame_count": 0,
        "vehicle_count": 0,
        "accident_alerts": 0,

        "accident_done": False,
        "normal_done": False,

        "accident_frame": None,
        "normal_frame": None,

        "accident_prediction": "Waiting",
        "normal_prediction": "Waiting",

        "accident_confidence": 0.0,
        "normal_confidence": 0.0,

        "accident_vehicle_count": 0,
        "normal_vehicle_count": 0,

        "accident_plates": [],
        "normal_plates": [],

        "all_plate_texts": [],

        "report_saved": False,

        "last_error": None,

        "last_alert_frame": -1
    }

    for key, value in values.items():
        st.session_state[key] = value


def save_report():

    if st.session_state.report_saved:
        return

    if not st.session_state.session_started:
        return

    report = {

        "Date & Time":
            st.session_state.session_started,

        "Accident Video":
            video_name(st.session_state.accident_path),

        "Non-Accident Video":
            video_name(st.session_state.normal_path),

        "Frames Processed":
            st.session_state.frame_count,

        "Vehicle Detections":
            st.session_state.vehicle_count,

        "Accident Alerts":
            st.session_state.accident_alerts,

        "Accident Prediction":
            st.session_state.accident_prediction,

        "Accident Confidence":
            round(st.session_state.accident_confidence, 2),

        "Non-Accident Prediction":
            st.session_state.normal_prediction,

        "Non-Accident Confidence":
            round(st.session_state.normal_confidence, 2),

        "Detected Number Plates":
            ", ".join(st.session_state.all_plate_texts)
    }

    st.session_state.history.append(report)

    st.session_state.report_saved = True


def start_detection():

    release_videos()

    reset_detection()

    accident_videos = get_video_files(ACCIDENT_DIR)

    normal_videos = get_video_files(NON_ACCIDENT_DIR)

    if not accident_videos or not normal_videos:

        raise FileNotFoundError(
            "Accident অথবা Non-Accident folder-এ video পাওয়া যায়নি।"
        )

    accident_path = random.choice(accident_videos)

    normal_path = random.choice(normal_videos)

    cnn = load_cnn_model()

    yolo = load_yolo_model()

    plate_model = load_plate_model()

    reader = load_ocr_reader() if plate_model else None

    cap_a = cv2.VideoCapture(accident_path)

    cap_n = cv2.VideoCapture(normal_path)

    if not cap_a.isOpened() or not cap_n.isOpened():

        cap_a.release()
        cap_n.release()

        raise RuntimeError(
            "Could not open one or both videos."
        )

    st.session_state.accident_path = accident_path
    st.session_state.normal_path = normal_path

    st.session_state.cap_accident = cap_a
    st.session_state.cap_normal = cap_n

    st.session_state.cnn_model = cnn
    st.session_state.yolo_model = yolo

    st.session_state.plate_model = plate_model
    st.session_state.ocr_reader = reader

    st.session_state.session_started = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.session_state.playing = True


def stop_detection():

    st.session_state.playing = False

    save_report()

    release_videos()


def replay_detection():

    accident_path = st.session_state.accident_path

    normal_path = st.session_state.normal_path

    if not accident_path or not normal_path:

        raise RuntimeError(
            "No previous video session available."
        )

    release_videos()

    cap_a = cv2.VideoCapture(accident_path)

    cap_n = cv2.VideoCapture(normal_path)

    if not cap_a.isOpened() or not cap_n.isOpened():

        cap_a.release()
        cap_n.release()

        raise RuntimeError(
            "Could not reopen videos."
        )

    st.session_state.cap_accident = cap_a
    st.session_state.cap_normal = cap_n

    reset_detection()

    st.session_state.playing = True


# =====================================================
# 12. PROCESS VIDEO FRAME
# =====================================================

def process_one_frame(

    cap_key,
    frame_key,
    prediction_key,
    confidence_key,
    vehicle_key,
    plates_key,
    done_key,
    frame_number,
    prediction_interval,
    yolo_interval,
    plate_interval

):

    cap = st.session_state.get(cap_key)

    if cap is None or st.session_state[done_key]:
        return

    ret, frame = cap.read()

    if not ret:

        st.session_state[done_key] = True

        return

    cnn = st.session_state.cnn_model

    yolo = st.session_state.yolo_model

    plate_model = st.session_state.plate_model

    reader = st.session_state.ocr_reader

    # CNN prediction
    if (
        st.session_state[prediction_key] == "Waiting"
        or frame_number % prediction_interval == 0
    ):

        prediction, confidence = predict_accident(
            cnn, frame
        )

        st.session_state[prediction_key] = prediction

        st.session_state[confidence_key] = confidence

    # Vehicle detection
    if (
        frame_number == 1
        or frame_number % yolo_interval == 0
    ):

        annotated, count = detect_vehicles(
            yolo, frame
        )

        st.session_state[vehicle_key] = count

    else:

        annotated = frame.copy()

    # Number plate detection and OCR
    if (
        plate_model is not None
        and (
            frame_number == 1
            or frame_number % plate_interval == 0
        )
    ):

        annotated, plates = detect_number_plates(
            plate_model,
            reader,
            annotated
        )

        st.session_state[plates_key] = plates

        for plate in plates:

            if plate != "Text Not Read":

                if plate not in st.session_state.all_plate_texts:

                    st.session_state.all_plate_texts.append(
                        plate
                    )

    # Accident alert
    if st.session_state[prediction_key] == "Accident":

        cv2.rectangle(
            annotated,
            (0, 0),
            (
                annotated.shape[1] - 1,
                annotated.shape[0] - 1
            ),
            (0, 0, 255),
            5
        )

        cv2.putText(
            annotated,
            "ACCIDENT ALERT",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    cv2.putText(
        annotated,
        (
            f"{st.session_state[prediction_key]} "
            f"{st.session_state[confidence_key]:.1f}%"
        ),
        (20, annotated.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    st.session_state[frame_key] = annotated


# =====================================================
# 13. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🚦 RoadSafe AI")

    st.caption(
        "AI Road Accident Monitoring System"
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Detection History",
            "System Information"
        ]
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.write(f"Device: {DEVICE.upper()}")

    if st.session_state.playing:

        st.success("LIVE MONITORING")

    else:

        st.info("STANDBY")


# =====================================================
# 14. MAIN HEADER
# =====================================================

st.markdown("""
<div class="main-header">

<h1>🚦 RoadSafe AI</h1>

<p>
AI-Powered Road Accident Detection,
Vehicle Monitoring & Number Plate Recognition
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 15. DASHBOARD
# =====================================================

if page == "Dashboard":

    accident_videos = get_video_files(ACCIDENT_DIR)

    normal_videos = get_video_files(NON_ACCIDENT_DIR)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Videos",
        len(accident_videos) + len(normal_videos)
    )

    c2.metric(
        "Accident Videos",
        len(accident_videos)
    )

    c3.metric(
        "Non-Accident Videos",
        len(normal_videos)
    )

    c4.metric(
        "Detection Sessions",
        len(st.session_state.history)
    )

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    s1, s2, s3 = st.columns(3)

    with s1:

        playback_speed = st.selectbox(
            "Playback Speed",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            index=3,
            format_func=lambda x: f"{x}x"
        )

    with s2:

        prediction_interval = st.selectbox(
            "CNN Prediction Interval",
            [1, 2, 3, 5, 10],
            index=2
        )

    with s3:

        yolo_interval = st.selectbox(
            "YOLO Detection Interval",
            [1, 2, 3, 5, 10],
            index=2
        )

    plate_interval = st.selectbox(
        "Number Plate Detection Interval",
        [1, 2, 3, 5, 10],
        index=2
    )

    brightness = st.slider(
        "☀️ Brightness Adjustment",
        0.5,
        2.0,
        1.0,
        0.1
    )

    zoom = st.slider(
        "🔍 Video Zoom",
        1.0,
        3.0,
        1.0,
        0.25
    )

    st.divider()

    st.markdown("### 🎮 Video Controls")

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:

        start_clicked = st.button(
            "▶ Start",
            use_container_width=True,
            disabled=st.session_state.playing
        )

    with b2:

        pause_clicked = st.button(
            "⏸ Pause",
            use_container_width=True,
            disabled=not st.session_state.playing
        )

    with b3:

        resume_clicked = st.button(
            "⏵ Resume",
            use_container_width=True,
            disabled=(
                st.session_state.playing
                or st.session_state.cap_accident is None
            )
        )

    with b4:

        stop_clicked = st.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=(
                st.session_state.cap_accident is None
                and st.session_state.cap_normal is None
            )
        )

    with b5:

        replay_clicked = st.button(
            "↻ Replay",
            use_container_width=True,
            disabled=(
                not st.session_state.accident_path
                or not st.session_state.normal_path
            )
        )

    if start_clicked:

        try:

            with st.spinner("Loading AI models..."):

                start_detection()

            st.rerun()

        except Exception as error:

            st.error(f"Start Error: {error}")

    if pause_clicked:

        st.session_state.playing = False

        st.rerun()

    if resume_clicked:

        st.session_state.playing = True

        st.rerun()

    if stop_clicked:

        stop_detection()

        st.rerun()

    if replay_clicked:

        try:

            replay_detection()

            st.rerun()

        except Exception as error:

            st.error(f"Replay Error: {error}")

    st.divider()

    st.markdown("### 📹 Live Video Monitoring")

    video_col1, video_col2 = st.columns(2)

    with video_col1:

        st.markdown(
            '<div class="accident-status">🔴 ACCIDENT VIDEO</div>',
            unsafe_allow_html=True
        )

        accident_name = st.empty()

        accident_placeholder = st.empty()

        accident_status = st.empty()

    with video_col2:

        st.markdown(
            '<div class="safe-status">🟢 NON-ACCIDENT VIDEO</div>',
            unsafe_allow_html=True
        )

        normal_name = st.empty()

        normal_placeholder = st.empty()

        normal_status = st.empty()

    accident_name.caption(
        video_name(st.session_state.accident_path)
    )

    normal_name.caption(
        video_name(st.session_state.normal_path)
    )

    st.divider()

    st.markdown("### 📊 Live Detection Analytics")

    m1, m2, m3, m4 = st.columns(4)

    frame_metric = m1.empty()
    vehicle_metric = m2.empty()
    alert_metric = m3.empty()
    plate_metric = m4.empty()

    # =================================================
    # 16. VIDEO FRAGMENT
    # =================================================

    @st.fragment(run_every=0.2)

    def video_monitor():

        if st.session_state.playing:

            try:

                frame_number = (
                    st.session_state.frame_count + 1
                )

                process_one_frame(
                    "cap_accident",
                    "accident_frame",
                    "accident_prediction",
                    "accident_confidence",
                    "accident_vehicle_count",
                    "accident_plates",
                    "accident_done",
                    frame_number,
                    prediction_interval,
                    yolo_interval,
                    plate_interval
                )

                process_one_frame(
                    "cap_normal",
                    "normal_frame",
                    "normal_prediction",
                    "normal_confidence",
                    "normal_vehicle_count",
                    "normal_plates",
                    "normal_done",
                    frame_number,
                    prediction_interval,
                    yolo_interval,
                    plate_interval
                )

                st.session_state.frame_count = frame_number

                st.session_state.vehicle_count = (
                    st.session_state.accident_vehicle_count
                    + st.session_state.normal_vehicle_count
                )

                if (
                    st.session_state.accident_prediction == "Accident"
                    and frame_number % prediction_interval == 0
                    and st.session_state.last_alert_frame != frame_number
                ):

                    st.session_state.accident_alerts += 1

                    st.session_state.last_alert_frame = frame_number

                if (
                    st.session_state.accident_done
                    and st.session_state.normal_done
                ):

                    st.session_state.playing = False

                    save_report()

                    release_videos()

            except Exception as error:

                st.session_state.last_error = str(error)

                st.session_state.playing = False

                save_report()

                release_videos()

        # ACCIDENT VIDEO
        if st.session_state.accident_frame is not None:

            frame = zoom_frame(
                st.session_state.accident_frame,
                zoom
            )

            frame = adjust_brightness(
                frame,
                brightness
            )

            accident_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

        # NORMAL VIDEO
        if st.session_state.normal_frame is not None:

            frame = zoom_frame(
                st.session_state.normal_frame,
                zoom
            )

            frame = adjust_brightness(
                frame,
                brightness
            )

            normal_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

        # ACCIDENT STATUS
        prediction = st.session_state.accident_prediction

        if prediction == "Accident":

            accident_status.error(
                f"ACCIDENT ALERT | "
                f"{st.session_state.accident_confidence:.1f}%"
            )

        elif prediction == "Non-Accident":

            accident_status.success(
                f"Non-Accident | "
                f"{st.session_state.accident_confidence:.1f}%"
            )

        # NORMAL STATUS
        prediction = st.session_state.normal_prediction

        if prediction == "Accident":

            normal_status.error(
                f"ACCIDENT ALERT | "
                f"{st.session_state.normal_confidence:.1f}%"
            )

        elif prediction == "Non-Accident":

            normal_status.success(
                f"Non-Accident | "
                f"{st.session_state.normal_confidence:.1f}%"
            )

        # ANALYTICS
        frame_metric.metric(
            "Frames Processed",
            st.session_state.frame_count
        )

        vehicle_metric.metric(
            "Vehicle Detections",
            st.session_state.vehicle_count
        )

        alert_metric.metric(
            "Accident Alerts",
            st.session_state.accident_alerts
        )

        plate_metric.metric(
            "Unique Plate Texts",
            len(st.session_state.all_plate_texts)
        )

        if st.session_state.last_error:

            st.error(st.session_state.last_error)

        if st.session_state.playing:

            st.caption("● Video Playing")

        elif st.session_state.session_started:

            st.caption("Video Paused, Stopped or Completed")

    video_monitor()

    # =================================================
    # 17. NUMBER PLATE RESULTS
    # =================================================

    st.divider()

    st.markdown("### 🔢 Number Plate Recognition Results")

    if not os.path.exists(PLATE_MODEL_PATH):

        st.warning(
            "License plate detection model is not installed. "
            "Add the trained model at models/license_plate_detector.pt "
            "to activate number plate detection."
        )

    elif st.session_state.all_plate_texts:

        plate_df = pd.DataFrame({

            "Serial": range(
                1,
                len(st.session_state.all_plate_texts) + 1
            ),

            "Detected Number Plate":
                st.session_state.all_plate_texts

        })

        st.dataframe(
            plate_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Number Plate CSV",
            data=plate_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="roadsafe_number_plates.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No number plate text recognized yet."
        )

    # LATEST REPORT
    if st.session_state.report_saved and st.session_state.history:

        st.divider()

        st.markdown("### 📄 Latest Detection Report")

        st.dataframe(
            pd.DataFrame([st.session_state.history[-1]]),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 18. DETECTION HISTORY
# =====================================================

elif page == "Detection History":

    st.markdown("## 📚 Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download CSV Report",
            data=csv_data,
            file_name="roadsafe_detection_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Detection History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("No detection history available.")


# =====================================================
# 19. SYSTEM INFORMATION
# =====================================================

elif page == "System Information":

    st.markdown("## ⚙️ System Information")

    st.markdown("""
    ### RoadSafe AI Architecture

    1. Video Input
    2. CNN Accident Classification
    3. YOLO Vehicle Detection
    4. License Plate Detection
    5. EasyOCR Number Plate Recognition
    6. OpenCV Video Processing
    7. Streamlit Dashboard
    8. Detection History and CSV Reports
    """)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### 🧠 CNN")

        st.write("Framework: PyTorch")

        st.write("Output: Accident / Non-Accident")

    with c2:

        st.markdown("### 🚘 YOLO")

        st.write("Model: YOLO11n")

        st.write("Vehicle Classes: Car, Motorcycle, Bus, Truck")

    st.divider()

    st.markdown("### 🔢 Number Plate System")

    st.write(
        "Plate Detector:",
        "Available" if os.path.exists(PLATE_MODEL_PATH)
        else "Model Not Installed"
    )

    st.write("OCR Engine: EasyOCR")

    st.write("OCR Language: English")

    st.write(f"Processing Device: {DEVICE.upper()}")

    st.write(
        f"CNN Model Available: {os.path.exists(MODEL_PATH)}"
    )


# =====================================================
# 20. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:12px;
    padding:12px;
">

RoadSafe AI | AI Accident Detection & Number Plate Recognition | 2026

</div>
""", unsafe_allow_html=True)
