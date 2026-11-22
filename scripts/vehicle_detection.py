import os
import random
import cv2
from ultralytics import YOLO

script_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_folder)

video_folder = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "accident"
)

model = YOLO("yolo11n.pt")

vehicle_classes = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

videos = []

for file in os.listdir(video_folder):

    if file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ):
        videos.append(
            os.path.join(video_folder, file)
        )

if not videos:
    print("No accident videos found!")
    exit()

video_path = random.choice(videos)

print("\nRandom accident video:")
print(os.path.basename(video_path))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.50,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            label = (
                f"{vehicle_classes[class_id]} "
                f"{confidence * 100:.1f}%"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "RoadSafe AI - Vehicle Detection",
        frame
    )

    key = cv2.waitKey(
        max(1, int(1000 / fps))
    ) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVehicle Detection Completed!")