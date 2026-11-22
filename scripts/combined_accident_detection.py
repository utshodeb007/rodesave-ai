import os
import random
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO


script_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_folder)

base_folder = os.path.join(
    project_root,
    "Data",
    "accident detection"
)

video_folder = os.path.join(
    base_folder,
    "accident"
)

cnn_model_path = os.path.join(
    base_folder,
    "accident_model.pth"
)


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
        return self.classifier(
            self.features(x)
        )


cnn = AccidentCNN()

cnn.load_state_dict(
    torch.load(
        cnn_model_path,
        map_location=device
    )
)

cnn = cnn.to(device)
cnn.eval()


yolo = YOLO("yolo11n.pt")


vehicle_classes = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


videos = []

for file in os.listdir(video_folder):

    if file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ):
        videos.append(
            os.path.join(
                video_folder,
                file
            )
        )


if not videos:
    print("No accident videos found!")
    exit()


video_path = random.choice(videos)

print("\n================================")
print("RANDOM ACCIDENT VIDEO")
print("================================")
print(
    os.path.basename(video_path)
)
print("================================")


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


frame_number = 0
last_prediction = "ACCIDENT"
last_confidence = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)


    with torch.no_grad():

        output = cnn(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        accident_probability = (
            probabilities[0][0].item()
            * 100
        )

        normal_probability = (
            probabilities[0][1].item()
            * 100
        )


    if accident_probability > normal_probability:

        last_prediction = "ACCIDENT"
        last_confidence = accident_probability

    else:

        last_prediction = "NON-ACCIDENT"
        last_confidence = normal_probability


    results = yolo(
        frame,
        conf=0.50,
        verbose=False
    )


    vehicle_count = 0


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


            vehicle_count += 1


            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            label = (
                f"{vehicle_classes[class_id]} "
                f"{confidence * 100:.0f}%"
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


    if last_prediction == "ACCIDENT":

        cv2.rectangle(
            frame,
            (0, 0),
            (
                frame.shape[1] - 1,
                frame.shape[0] - 1
            ),
            (0, 0, 255),
            5
        )

        cv2.putText(
            frame,
            f"ACCIDENT DETECTED "
            f"{last_confidence:.1f}%",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            f"NON-ACCIDENT "
            f"{last_confidence:.1f}%",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )


    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "RoadSafe AI - Combined Detection",
        frame
    )


    key = cv2.waitKey(
        max(1, int(1000 / fps))
    ) & 0xFF


    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\n================================")
print("COMBINED DETECTION COMPLETED")
print("================================")