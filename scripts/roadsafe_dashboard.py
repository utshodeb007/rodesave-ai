
import os
import random
import re
import io
import math
import struct
import wave
from datetime import datetime

import cv2
import easyocr
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
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
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")

YOLO_MODEL_PATH = "yolo11n.pt"

SUPPORTED_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm")

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
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



@st.cache_data
def create_siren_audio():
    """Create a short, original two-tone alert sound as WAV bytes."""
    sample_rate = 22050
    duration = 1.6
    samples = []

    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        phase = int(t / 0.4) % 2
        frequency = 880 if phase == 0 else 660
        pulse = 0.75 if (t % 0.4) < 0.32 else 0.0
        fade = min(1.0, t * 12, (duration - t) * 12)
        value = int(
            15000 * pulse * fade *
            math.sin(2 * math.pi * frequency * t)
        )
        samples.append(struct.pack("<h", value))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(b"".join(samples))

    return buffer.getvalue()


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1120;
    color: #E2E8F0;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1650px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #26364D;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.main-header {
    background: linear-gradient(120deg, #101827, #142B38, #153D46);
    border: 1px solid #28505A;
    padding: 28px;
    border-radius: 17px;
    margin-bottom: 22px;
    box-shadow: 0 8px 30px #00000035;
}

.main-header h1 {
    color: #F8FAFC;
    font-size: 32px;
    font-weight: 800;
}

.main-header p {
    color: #A5C3C5;
}

.section-heading {
    color: #F1F5F9;
    font-size: 20px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172337, #111C2E);
    border: 1px solid #293B53;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
    font-weight: 800;
}

div.stButton > button {
    background: #153D46;
    color: #F0FDFA;
    border: 1px solid #287D7A;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
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
    padding: 14px;
    border-radius: 10px;
    font-weight: 700;
}

.safe-status {
    background: #123528;
    border: 1px solid #22C55E;
    color: #86EFAC;
    padding: 14px;
    border-radius: 10px;
    font-weight: 700;
}

div[data-testid="stProgressBar"] > div > div {
    background: #14B8A6;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293B53;
    border-radius: 12px;
}

h1, h2, h3 {
    color: #F1F5F9;
}

hr {
    border-color: #293B53;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. CNN MODEL
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
# 5. LOAD AI MODELS
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

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
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
    return easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))


def detect_number_plates(plate_model, reader, frame):
    annotated = frame.copy()
    detected_plates = []
    if plate_model is None or reader is None:
        return annotated, detected_plates

    results = plate_model.predict(frame, conf=0.25, verbose=False)
    height, width = frame.shape[:2]
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            words = reader.readtext(gray, detail=0, paragraph=False)
            plate_text = " ".join(str(word).strip() for word in words if str(word).strip())
            plate_text = re.sub(r"[^A-Za-z0-9 -]", "", plate_text).strip() or "Text Not Read"
            detected_plates.append(plate_text)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 3)
            cv2.putText(annotated, plate_text[:35], (x1, max(y1 - 12, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
    return annotated, detected_plates


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

    if not path:
        return "No video selected"

    return os.path.basename(path)


def zoom_frame(frame, zoom):

    if frame is None or zoom <= 1:
        return frame

    height, width = frame.shape[:2]

    crop_width = int(width / zoom)
    crop_height = int(height / zoom)

    cx = width // 2
    cy = height // 2

    x1 = max(0, cx - crop_width // 2)
    y1 = max(0, cy - crop_height // 2)

    x2 = min(width, x1 + crop_width)
    y2 = min(height, y1 + crop_height)

    cropped = frame[y1:y2, x1:x2]

    return cv2.resize(cropped, (width, height))


# =====================================================
# 7. CNN ACCIDENT PREDICTION
# =====================================================

def predict_accident(model, frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    tensor = IMAGE_TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    label = CLASS_NAMES[prediction.item()]
    confidence = float(confidence.item()) * 100

    return label, confidence


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
# 9. SESSION STATE
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
    "accident_plate_texts": [],
    "normal_plate_texts": [],
    "frame_count": 0,
    "vehicle_count": 0,
    "accident_alerts": 0,
    "siren_event": 0,
    "siren_played_event": 0,
    "sound_alert_enabled": True,
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
    "session_started": None,
    "report_saved": False,
    "last_error": None
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 10. VIDEO CONTROL FUNCTIONS
# =====================================================

def release_videos():

    for key in ["cap_accident", "cap_normal"]:

        cap = st.session_state.get(key)

        if cap is not None:
            cap.release()

        st.session_state[key] = None


def reset_detection_state():

    keys = {
        "frame_count": 0,
        "vehicle_count": 0,
        "accident_alerts": 0,
        "siren_event": 0,
        "siren_played_event": 0,
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
        "accident_plate_texts": [],
        "normal_plate_texts": [],
        "report_saved": False,
        "last_error": None
    }

    for key, value in keys.items():
        st.session_state[key] = value


def save_report():

    if st.session_state.report_saved:
        return

    if not st.session_state.session_started:
        return

    report = {
        "Date & Time": st.session_state.session_started,
        "Accident Video": video_name(st.session_state.accident_path),
        "Non-Accident Video": video_name(st.session_state.normal_path),
        "Frames Processed": st.session_state.frame_count,
        "Vehicle Detections": st.session_state.vehicle_count,
        "Accident Alerts": st.session_state.accident_alerts,
        "Accident Prediction": st.session_state.accident_prediction,
        "Accident Confidence": round(
            st.session_state.accident_confidence, 2
        ),
        "Normal Prediction": st.session_state.normal_prediction,
        "Normal Confidence": round(
            st.session_state.normal_confidence, 2
        )
    }

    st.session_state.history.append(report)
    st.session_state.report_saved = True


def start_detection():

    release_videos()
    reset_detection_state()

    accident_videos = get_video_files(ACCIDENT_DIR)
    normal_videos = get_video_files(NON_ACCIDENT_DIR)

    if not accident_videos or not normal_videos:
        raise FileNotFoundError(
            "Accident অথবা Non-Accident folder-এ ভিডিও পাওয়া যায়নি।"
        )

    accident_path = random.choice(accident_videos)
    normal_path = random.choice(normal_videos)

    cnn = load_cnn_model()
    yolo = load_yolo_model()
    plate_model = load_plate_model()
    ocr_reader = load_ocr_reader() if plate_model is not None else None

    cap_a = cv2.VideoCapture(accident_path)
    cap_n = cv2.VideoCapture(normal_path)

    if not cap_a.isOpened() or not cap_n.isOpened():

        cap_a.release()
        cap_n.release()

        raise RuntimeError("Could not open one or both videos.")

    st.session_state.accident_path = accident_path
    st.session_state.normal_path = normal_path

    st.session_state.cap_accident = cap_a
    st.session_state.cap_normal = cap_n

    st.session_state.cnn_model = cnn
    st.session_state.yolo_model = yolo
    st.session_state.plate_model = plate_model
    st.session_state.ocr_reader = ocr_reader

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
        raise RuntimeError("No previous video session available.")

    release_videos()

    cap_a = cv2.VideoCapture(accident_path)
    cap_n = cv2.VideoCapture(normal_path)

    if not cap_a.isOpened() or not cap_n.isOpened():

        cap_a.release()
        cap_n.release()

        raise RuntimeError("Could not reopen videos.")

    st.session_state.cap_accident = cap_a
    st.session_state.cap_normal = cap_n

    reset_detection_state()
    st.session_state.playing = True


# =====================================================
# 11. PROCESS VIDEO FRAME
# =====================================================

def process_one_frame(
    cap_key,
    frame_key,
    prediction_key,
    confidence_key,
    vehicle_key,
    done_key,
    frame_number,
    prediction_interval,
    yolo_interval
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

    if (
        st.session_state[prediction_key] == "Waiting"
        or frame_number % prediction_interval == 0
    ):

        previous_prediction = st.session_state[prediction_key]
        prediction, confidence = predict_accident(cnn, frame)

        st.session_state[prediction_key] = prediction
        st.session_state[confidence_key] = confidence

        # Trigger sound only when a stream changes into Accident.
        if prediction == "Accident" and previous_prediction != "Accident":
            st.session_state.siren_event += 1

    if frame_number == 1 or frame_number % yolo_interval == 0:
        annotated, count = detect_vehicles(yolo, frame)
        st.session_state[vehicle_key] = count
        annotated, plate_texts = detect_number_plates(
            st.session_state.get("plate_model"),
            st.session_state.get("ocr_reader"),
            annotated
        )
        plate_key = "accident_plate_texts" if cap_key == "cap_accident" else "normal_plate_texts"
        for plate_text in plate_texts:
            if plate_text not in st.session_state[plate_key]:
                st.session_state[plate_key].append(plate_text)
    else:
        annotated = frame.copy()

    if st.session_state[prediction_key] == "Accident":

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1] - 1, annotated.shape[0] - 1),
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
# 12. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🚦 RoadSafe AI")
    st.caption("Advanced Accident Monitoring System")

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
    st.write(f"Processing Device: {DEVICE.upper()}")

    if st.session_state.playing:
        st.success("LIVE MONITORING")
    else:
        st.info("STANDBY")

    st.divider()
    st.caption("RoadSafe AI | 2026")


# =====================================================
# 13. MAIN HEADER
# =====================================================

st.markdown("""
<div class="main-header">
    <h1>🚦 RoadSafe AI</h1>
    <p>
        AI-Powered Road Accident Detection & Vehicle Monitoring System
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 14. DASHBOARD
# =====================================================

if page == "Dashboard":

    accident_videos = get_video_files(ACCIDENT_DIR)
    normal_videos = get_video_files(NON_ACCIDENT_DIR)

    # Number Plate model status: distinguish file presence from actual loading.
    if not os.path.exists(PLATE_MODEL_PATH):
        st.warning(
            "Number Plate Model Missing. "
            "Accident Detection and Vehicle Detection will continue normally."
        )
    elif (
        st.session_state.get("plate_model") is not None
        and st.session_state.get("ocr_reader") is not None
    ):
        st.success("Number Plate Detection Active — Model + EasyOCR Loaded.")
    elif st.session_state.get("plate_model") is not None:
        st.warning(
            "Number Plate Model Loaded, but OCR is not available. "
            "Check EasyOCR installation or loading errors."
        )
    elif st.session_state.get("session_started"):
        st.warning(
            "Number Plate Model is not loaded in the current session. "
            "Press Start or Replay to load it."
        )
    else:
        st.info(
            "Number Plate model file found. "
            "Press Start to load the model and EasyOCR."
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Videos",
        len(accident_videos) + len(normal_videos)
    )

    col2.metric(
        "Accident Videos",
        len(accident_videos)
    )

    col3.metric(
        "Non-Accident Videos",
        len(normal_videos)
    )

    col4.metric(
        "Detection Sessions",
        len(st.session_state.history)
    )

    st.divider()

    st.markdown(
        '<div class="section-heading">⚙️ Detection Settings</div>',
        unsafe_allow_html=True
    )

    setting1, setting2, setting3 = st.columns(3)

    with setting1:
        playback_speed = st.selectbox(
            "Playback Speed",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            index=3,
            format_func=lambda x: f"{x}x"
        )

    with setting2:
        prediction_interval = st.selectbox(
            "CNN Prediction Interval",
            [1, 2, 3, 5, 10],
            index=2,
            format_func=lambda x: f"Every {x} frames"
        )

    with setting3:
        yolo_interval = st.selectbox(
            "YOLO Detection Interval",
            [1, 2, 3, 5, 10],
            index=2,
            format_func=lambda x: f"Every {x} frames"
        )

    zoom_value = st.slider(
        "🔍 Video Zoom",
        1.0,
        3.0,
        1.0,
        0.25
    )

    brightness_value = st.slider(
        "☀️ Video Brightness",
        -60,
        60,
        0,
        5,
        help="0 = original brightness; negative = darker; positive = brighter."
    )

    st.session_state.sound_alert_enabled = st.toggle(
        "🔊 Enable Accident Siren",
        value=st.session_state.sound_alert_enabled,
        help="Play a short alert sound when a video is newly classified as Accident."
    )

    st.divider()

    st.markdown(
        '<div class="section-heading">🎮 Video Controls</div>',
        unsafe_allow_html=True
    )

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
            st.error(f"Start failed: {error}")

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
            st.error(f"Replay failed: {error}")

    # VIDEO PROCESSING + LIVE MONITORING
    # Streamlit fragments update only this monitoring area.
    # This avoids rerunning the entire dashboard for every video frame.

    refresh_seconds = max(
        0.08,
        min(1.0 / (12 * playback_speed), 1.0)
    )

    if not hasattr(st, "fragment"):
        st.error(
            "এই dashboard-এর smooth playback-এর জন্য Streamlit 1.37 বা newer "
            "প্রয়োজন। Terminal-এ চালাও: python3 -m pip install --upgrade streamlit"
        )
    else:

        @st.fragment(
            run_every=refresh_seconds if st.session_state.playing else None
        )
        def render_live_monitoring():

            playback_ended = False

            if st.session_state.playing:
                st.session_state.frame_count += 1
                frame_number = st.session_state.frame_count

                try:
                    process_one_frame(
                        "cap_accident",
                        "accident_frame",
                        "accident_prediction",
                        "accident_confidence",
                        "accident_vehicle_count",
                        "accident_done",
                        frame_number,
                        prediction_interval,
                        yolo_interval
                    )

                    process_one_frame(
                        "cap_normal",
                        "normal_frame",
                        "normal_prediction",
                        "normal_confidence",
                        "normal_vehicle_count",
                        "normal_done",
                        frame_number,
                        prediction_interval,
                        yolo_interval
                    )

                    st.session_state.vehicle_count = (
                        st.session_state.accident_vehicle_count
                        + st.session_state.normal_vehicle_count
                    )

                    if (
                        st.session_state.accident_prediction == "Accident"
                        and frame_number % prediction_interval == 0
                    ):
                        st.session_state.accident_alerts += 1

                    if (
                        st.session_state.accident_done
                        and st.session_state.normal_done
                    ):
                        st.session_state.playing = False
                        save_report()
                        release_videos()
                        playback_ended = True

                except Exception as error:
                    st.session_state.last_error = str(error)
                    st.session_state.playing = False
                    save_report()
                    release_videos()
                    playback_ended = True

            # Stop the fragment's timed reruns after the video ends.
            if playback_ended:
                st.rerun()

            # Play one short siren for each new Accident transition.
            current_siren_event = st.session_state.siren_event
            if (
                st.session_state.sound_alert_enabled
                and current_siren_event > st.session_state.siren_played_event
            ):
                st.warning("🚨 ACCIDENT DETECTED — SOUND ALERT")
                st.audio(
                    create_siren_audio(),
                    format="audio/wav",
                    autoplay=True
                )
                st.session_state.siren_played_event = current_siren_event

            # VIDEO PANELS
            st.divider()
            st.markdown(
                '<div class="section-heading">📹 Live Video Monitoring</div>',
                unsafe_allow_html=True
            )

            video_col1, video_col2 = st.columns(2)

            with video_col1:
                st.markdown(
                    '<div class="accident-status">🔴 ACCIDENT VIDEO</div>',
                    unsafe_allow_html=True
                )
                st.caption(video_name(st.session_state.accident_path))

                if st.session_state.accident_frame is not None:
                    frame = zoom_frame(
                        st.session_state.accident_frame,
                        zoom_value
                    )
                    frame = cv2.convertScaleAbs(
                        frame, alpha=1.0, beta=brightness_value
                    )
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                else:
                    st.info("Accident video preview will appear here.")

                prediction = st.session_state.accident_prediction
                if prediction == "Accident":
                    st.error(
                        f"ACCIDENT ALERT | "
                        f"{st.session_state.accident_confidence:.1f}%"
                    )
                elif prediction == "Non-Accident":
                    st.success(
                        f"Non-Accident | "
                        f"{st.session_state.accident_confidence:.1f}%"
                    )
                else:
                    st.info("Waiting for prediction.")

            with video_col2:
                st.markdown(
                    '<div class="safe-status">🟢 NON-ACCIDENT VIDEO</div>',
                    unsafe_allow_html=True
                )
                st.caption(video_name(st.session_state.normal_path))

                if st.session_state.normal_frame is not None:
                    frame = zoom_frame(
                        st.session_state.normal_frame,
                        zoom_value
                    )
                    frame = cv2.convertScaleAbs(
                        frame, alpha=1.0, beta=brightness_value
                    )
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                else:
                    st.info("Non-Accident video preview will appear here.")

                prediction = st.session_state.normal_prediction
                if prediction == "Accident":
                    st.error(
                        f"ACCIDENT ALERT | "
                        f"{st.session_state.normal_confidence:.1f}%"
                    )
                elif prediction == "Non-Accident":
                    st.success(
                        f"Non-Accident | "
                        f"{st.session_state.normal_confidence:.1f}%"
                    )
                else:
                    st.info("Waiting for prediction.")

            # LIVE ANALYTICS — refreshed inside this fragment only
            st.divider()
            st.markdown(
                '<div class="section-heading">📊 Live Detection Analytics</div>',
                unsafe_allow_html=True
            )

            live1, live2, live3, live4 = st.columns(4)
            live1.metric("Frames Processed", st.session_state.frame_count)
            live2.metric("Vehicle Detections", st.session_state.vehicle_count)
            live3.metric("Accident Alerts", st.session_state.accident_alerts)
            live4.metric("Playback Speed", f"{playback_speed}x")

            if st.session_state.playing:
                st.info("Video is playing.")
            elif (
                st.session_state.cap_accident is not None
                or st.session_state.cap_normal is not None
            ):
                st.warning("Video paused. Press Resume to continue.")
            elif st.session_state.session_started:
                st.success("Detection stopped or completed.")

            if st.session_state.last_error:
                st.error(st.session_state.last_error)

            st.markdown("### 🔎 Number Plate OCR Results")
            plate_col1, plate_col2 = st.columns(2)
            with plate_col1:
                st.markdown("**Accident Video Plates**")
                if st.session_state.accident_plate_texts:
                    for plate in st.session_state.accident_plate_texts:
                        st.code(plate)
                else:
                    st.caption("No plate text recognized yet.")
            with plate_col2:
                st.markdown("**Non-Accident Video Plates**")
                if st.session_state.normal_plate_texts:
                    for plate in st.session_state.normal_plate_texts:
                        st.code(plate)
                else:
                    st.caption("No plate text recognized yet.")

            if st.session_state.report_saved and st.session_state.history:
                st.markdown("### 📄 Latest Detection Report")
                st.dataframe(
                    pd.DataFrame([st.session_state.history[-1]]),
                    use_container_width=True,
                    hide_index=True
                )

        render_live_monitoring()


# =====================================================
# 15. DETECTION HISTORY
# =====================================================

elif page == "Detection History":

    st.markdown("## 📚 Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

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

        st.info(
            "No detection history available. "
            "Start a detection session first."
        )


# =====================================================
# 16. SYSTEM INFORMATION
# =====================================================

elif page == "System Information":

    st.markdown("## ⚙️ System Information")

    st.markdown("""
    ### RoadSafe AI Architecture

    1. Video Input: Accident and Non-Accident datasets.
    2. CNN Model: Accident classification using PyTorch.
    3. YOLO Model: Vehicle detection and bounding boxes.
    4. OpenCV: Video frame processing.
    5. Streamlit: Interactive web dashboard.
    6. Report Module: Detection history and CSV export.
    7. Optional Number Plate Model: Plate detection when trained weights are available.
    """)

    info1, info2 = st.columns(2)

    with info1:

        st.markdown("### 🧠 CNN Accident Classifier")

        st.write("Input: 128 × 128 RGB image")
        st.write("Output: Accident / Non-Accident")
        st.write("Framework: PyTorch")

    with info2:

        st.markdown("### 🚘 YOLO Vehicle Detector")

        st.write("Model: YOLO11n")
        st.write("Classes: Car, Motorcycle, Bus, Truck")
        st.write("Framework: Ultralytics")

    st.divider()

    st.markdown("### 📁 Dataset Status")

    st.write(
        f"Accident Videos: {len(get_video_files(ACCIDENT_DIR))}"
    )

    st.write(
        f"Non-Accident Videos: {len(get_video_files(NON_ACCIDENT_DIR))}"
    )

    st.write(
        f"CNN Model Available: "
        f"{'Yes' if os.path.exists(MODEL_PATH) else 'No'}"
    )

    st.write(
        f"Number Plate Model Available: "
        f"{'Yes' if os.path.exists(PLATE_MODEL_PATH) else 'No'}"
    )

    st.write(f"Processing Device: {DEVICE.upper()}")


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:12px;
    padding:12px;
">
    RoadSafe AI | Advanced Accident Detection & Monitoring System | 2026
</div>
""", unsafe_allow_html=True)
