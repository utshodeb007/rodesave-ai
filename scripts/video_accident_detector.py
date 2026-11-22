
import os
import cv2
import time
import random
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image


# ==========================================
# 1. PROJECT PATH
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BASE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "accident detection"
)

MODEL_PATH = os.path.join(
    BASE_FOLDER,
    "accident_model.pth"
)

ACCIDENT_FOLDER = os.path.join(
    BASE_FOLDER,
    "accident"
)

NON_ACCIDENT_FOLDER = os.path.join(
    BASE_FOLDER,
    "non_accident"
)


# ==========================================
# 2. DEVICE CONFIGURATION
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# 3. CNN MODEL
# Compatible with saved model layer names:
# features.0, features.3, features.6
# classifier.1, classifier.4
# ==========================================

class AccidentCNN(nn.Module):

    def __init__(self):
        super(AccidentCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = AccidentCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

model.eval()

print("CNN Model loaded successfully!")


# ==========================================
# 5. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor()

])

CLASS_NAMES = [
    "ACCIDENT",
    "NON-ACCIDENT"
]


# ==========================================
# 6. VIDEO FILE SELECTION
# ==========================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg"
)


def get_random_video(folder):

    if not os.path.isdir(folder):

        raise FileNotFoundError(
            f"Video folder not found: {folder}"
        )

    videos = [

        os.path.join(folder, file)

        for file in os.listdir(folder)

        if file.lower().endswith(VIDEO_EXTENSIONS)

    ]

    if not videos:

        raise FileNotFoundError(
            f"No video files found in: {folder}"
        )

    return random.choice(videos)


# Select one video from each folder

accident_video = get_random_video(
    ACCIDENT_FOLDER
)

non_accident_video = get_random_video(
    NON_ACCIDENT_FOLDER
)

print("\nSelected Videos:")

print(
    "Accident Video:",
    os.path.basename(accident_video)
)

print(
    "Non-Accident Video:",
    os.path.basename(non_accident_video)
)


# ==========================================
# 7. OPEN VIDEO FUNCTION
# ==========================================

def open_videos():

    cap1 = cv2.VideoCapture(accident_video)

    cap2 = cv2.VideoCapture(non_accident_video)

    if not cap1.isOpened():

        cap1.release()
        cap2.release()

        raise RuntimeError(
            "Cannot open Accident video."
        )

    if not cap2.isOpened():

        cap1.release()
        cap2.release()

        raise RuntimeError(
            "Cannot open Non-Accident video."
        )

    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    if fps1 <= 0:
        fps1 = 30.0

    if fps2 <= 0:
        fps2 = 30.0

    return cap1, cap2, fps1, fps2


# ==========================================
# 8. OPEN BOTH VIDEOS
# ==========================================

cap_accident, cap_non_accident, fps_accident, fps_non_accident = open_videos()


# ==========================================
# 9. CREATE TWO WINDOWS
# ==========================================

WINDOW_ACCIDENT = "RoadSafe AI - Accident Video"

WINDOW_NON_ACCIDENT = "RoadSafe AI - Non-Accident Video"

cv2.namedWindow(
    WINDOW_ACCIDENT,
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    WINDOW_NON_ACCIDENT,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_ACCIDENT,
    700,
    450
)

cv2.resizeWindow(
    WINDOW_NON_ACCIDENT,
    700,
    450
)


# ==========================================
# 10. CNN PREDICTION FUNCTION
# ==========================================

def predict_frame(frame):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(image)

    image_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    prediction = CLASS_NAMES[
        predicted_class.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    return prediction, confidence_value


# ==========================================
# 11. DRAW PREDICTION
# ==========================================

def draw_prediction(
    frame,
    prediction,
    confidence,
    title,
    speed,
    status
):

    display_frame = cv2.resize(
        frame,
        (700, 400)
    )

    if prediction == "ACCIDENT":

        color = (0, 0, 255)

        cv2.rectangle(
            display_frame,
            (0, 0),
            (699, 399),
            color,
            8
        )

        cv2.putText(
            display_frame,
            "!!! ACCIDENT DETECTED !!!",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    elif prediction == "NON-ACCIDENT":

        color = (0, 255, 0)

        cv2.putText(
            display_frame,
            "NO ACCIDENT DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    else:

        color = (0, 255, 255)

        cv2.putText(
            display_frame,
            "ANALYZING VIDEO...",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.putText(
        display_frame,
        f"Prediction: {prediction}",
        (20, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        display_frame,
        f"Confidence: {confidence:.2f}%",
        (20, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Speed: {speed:.2f}x | {status}",
        (20, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        title,
        (440, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return display_frame


# ==========================================
# 12. PLAYBACK SETTINGS
# ==========================================

PLAYBACK_SPEED = 0.5

MIN_SPEED = 0.25

MAX_SPEED = 2.0

PREDICTION_INTERVAL = 10

paused = False

video_accident_active = True

video_non_accident_active = True

frame_count_accident = 0

frame_count_non_accident = 0

processed_accident = 0

processed_non_accident = 0

accident_predictions = 0

non_accident_predictions = 0

last_prediction_accident = "ANALYZING..."

last_prediction_non_accident = "ANALYZING..."

confidence_accident = 0.0

confidence_non_accident = 0.0

last_frame_accident = None

last_frame_non_accident = None

next_accident_frame = time.monotonic()

next_non_accident_frame = time.monotonic()


# ==========================================
# 13. REPLAY FUNCTION
# ==========================================

def reset_videos():

    global cap_accident
    global cap_non_accident

    global fps_accident
    global fps_non_accident

    global frame_count_accident
    global frame_count_non_accident

    global processed_accident
    global processed_non_accident

    global accident_predictions
    global non_accident_predictions

    global last_prediction_accident
    global last_prediction_non_accident

    global confidence_accident
    global confidence_non_accident

    global last_frame_accident
    global last_frame_non_accident

    global video_accident_active
    global video_non_accident_active

    global next_accident_frame
    global next_non_accident_frame

    global paused

    cap_accident.release()

    cap_non_accident.release()

    (
        cap_accident,
        cap_non_accident,
        fps_accident,
        fps_non_accident
    ) = open_videos()

    frame_count_accident = 0

    frame_count_non_accident = 0

    processed_accident = 0

    processed_non_accident = 0

    accident_predictions = 0

    non_accident_predictions = 0

    last_prediction_accident = "ANALYZING..."

    last_prediction_non_accident = "ANALYZING..."

    confidence_accident = 0.0

    confidence_non_accident = 0.0

    last_frame_accident = None

    last_frame_non_accident = None

    video_accident_active = True

    video_non_accident_active = True

    paused = False

    next_accident_frame = time.monotonic()

    next_non_accident_frame = time.monotonic()

    print("\nBoth videos restarted!")


# ==========================================
# 14. MAIN VIDEO PROCESSING
# ==========================================

print("\n====================================")

print(" ROADSAFE AI - DUAL VIDEO DETECTION")

print("====================================")

print("Default Speed: 0.5x")

print("SPACE = Pause / Resume")

print("+ = Increase Speed")

print("- = Decrease Speed")

print("R = Replay Both Videos")

print("Q = Exit")

print("====================================")


try:

    while True:

        current_time = time.monotonic()

        # ==================================
        # ACCIDENT VIDEO
        # ==================================

        if video_accident_active and not paused:

            if current_time >= next_accident_frame:

                ret1, frame1 = cap_accident.read()

                if ret1:

                    frame_count_accident += 1

                    processed_accident += 1

                    # CNN prediction

                    if (
                        frame_count_accident
                        % PREDICTION_INTERVAL == 0
                    ):

                        (
                            last_prediction_accident,
                            confidence_accident
                        ) = predict_frame(frame1)

                        if last_prediction_accident == "ACCIDENT":

                            accident_predictions += 1

                        print(
                            f"[ACCIDENT] "
                            f"Frame: {frame_count_accident} | "
                            f"Prediction: {last_prediction_accident} | "
                            f"Confidence: {confidence_accident:.2f}%"
                        )

                    display1 = draw_prediction(
                        frame1,
                        last_prediction_accident,
                        confidence_accident,
                        "ACCIDENT VIDEO",
                        PLAYBACK_SPEED,
                        "PLAYING"
                    )

                    last_frame_accident = display1.copy()

                    cv2.imshow(
                        WINDOW_ACCIDENT,
                        display1
                    )

                    frame_delay = (
                        1.0 / fps_accident
                    ) / PLAYBACK_SPEED

                    next_accident_frame = (
                        time.monotonic() + frame_delay
                    )

                else:

                    video_accident_active = False

                    print("\nAccident Video Completed!")

                    print("Last frame will remain visible.")


        # ==================================
        # NON-ACCIDENT VIDEO
        # ==================================

        if video_non_accident_active and not paused:

            if current_time >= next_non_accident_frame:

                ret2, frame2 = cap_non_accident.read()

                if ret2:

                    frame_count_non_accident += 1

                    processed_non_accident += 1

                    # CNN prediction

                    if (
                        frame_count_non_accident
                        % PREDICTION_INTERVAL == 0
                    ):

                        (
                            last_prediction_non_accident,
                            confidence_non_accident
                        ) = predict_frame(frame2)

                        if last_prediction_non_accident == "NON-ACCIDENT":

                            non_accident_predictions += 1

                        print(
                            f"[NON-ACCIDENT] "
                            f"Frame: {frame_count_non_accident} | "
                            f"Prediction: {last_prediction_non_accident} | "
                            f"Confidence: {confidence_non_accident:.2f}%"
                        )

                    display2 = draw_prediction(
                        frame2,
                        last_prediction_non_accident,
                        confidence_non_accident,
                        "NON-ACCIDENT VIDEO",
                        PLAYBACK_SPEED,
                        "PLAYING"
                    )

                    last_frame_non_accident = display2.copy()

                    cv2.imshow(
                        WINDOW_NON_ACCIDENT,
                        display2
                    )

                    frame_delay = (
                        1.0 / fps_non_accident
                    ) / PLAYBACK_SPEED

                    next_non_accident_frame = (
                        time.monotonic() + frame_delay
                    )

                else:

                    video_non_accident_active = False

                    print("\nNon-Accident Video Completed!")

                    print("Last frame will remain visible.")


        # ==================================
        # KEYBOARD CONTROL
        # ==================================

        key = cv2.waitKey(10) & 0xFF

        # EXIT
        if key == ord("q"):

            print("\nRoadSafe AI stopped.")

            break

        # PAUSE / RESUME
        elif key == ord(" "):

            paused = not paused

            if paused:

                print("\nVideos Paused.")

            else:

                next_accident_frame = time.monotonic()

                next_non_accident_frame = time.monotonic()

                print("\nVideos Resumed.")

        # INCREASE SPEED
        elif key in (ord("+"), ord("=")):

            PLAYBACK_SPEED = min(
                MAX_SPEED,
                PLAYBACK_SPEED + 0.25
            )

            next_accident_frame = time.monotonic()

            next_non_accident_frame = time.monotonic()

            print(
                f"\nPlayback Speed: {PLAYBACK_SPEED:.2f}x"
            )

        # DECREASE SPEED
        elif key in (ord("-"), ord("_")):

            PLAYBACK_SPEED = max(
                MIN_SPEED,
                PLAYBACK_SPEED - 0.25
            )

            next_accident_frame = time.monotonic()

            next_non_accident_frame = time.monotonic()

            print(
                f"\nPlayback Speed: {PLAYBACK_SPEED:.2f}x"
            )

        # REPLAY BOTH VIDEOS
        elif key == ord("r"):

            print("\nReplaying both videos...")

            reset_videos()


finally:

    # ======================================
    # 15. RELEASE RESOURCES
    # ======================================

    cap_accident.release()

    cap_non_accident.release()

    cv2.destroyAllWindows()


    # ======================================
    # 16. FINAL RESULTS
    # ======================================

    print("\n====================================")

    print("       ROADSAFE AI RESULTS")

    print("====================================")

    print("\nACCIDENT VIDEO")

    print(
        "Total Frames Processed:",
        processed_accident
    )

    print(
        "Accident Predictions:",
        accident_predictions
    )

    print("\nNON-ACCIDENT VIDEO")

    print(
        "Total Frames Processed:",
        processed_non_accident
    )

    print(
        "Non-Accident Predictions:",
        non_accident_predictions
    )

    print("\n====================================")

    print("       TESTING COMPLETED")

    print("====================================")
