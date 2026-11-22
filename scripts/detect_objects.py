from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo26n.pt")

# Main folders
frames_folder = "data/frames"
output_folder = "data/detected_frames"

# Objects we want to detect
allowed_classes = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Minimum confidence
confidence_threshold = 0.50

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Find all video/frame folders
folders = [
    folder for folder in os.listdir(frames_folder)
    if not folder.startswith(".")
]

print("Frame folders found:", len(folders))

if len(folders) == 0:
    print("No frame folders found!")
    exit()

# Process every frame folder
for folder in folders:

    folder_path = os.path.join(frames_folder, folder)

    if not os.path.isdir(folder_path):
        continue

    print("\nProcessing folder:", folder)

    # Create separate output folder
    video_output_folder = os.path.join(
        output_folder,
        folder
    )

    os.makedirs(video_output_folder, exist_ok=True)

    # Find images
    images = [
        image for image in os.listdir(folder_path)
        if image.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print("Frames found:", len(images))

    # Process every frame
    for count, image in enumerate(images, start=1):

        image_path = os.path.join(
            folder_path,
            image
        )

        frame = cv2.imread(image_path)

        if frame is None:
            print("Could not read:", image)
            continue

        # YOLO detection
        results = model(
            image_path,
            conf=confidence_threshold,
            verbose=False
        )

        result = results[0]

        # Draw selected detections
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in allowed_classes:
                continue

            label = allowed_classes[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Save detected frame
        output_path = os.path.join(
            video_output_folder,
            image
        )

        cv2.imwrite(
            output_path,
            frame
        )

        # Progress
        if count % 10 == 0:
            print(
                f"Processed {count}/{len(images)} frames"
            )

    print(
        "Completed:",
        folder
    )

print("\n================================")
print("ALL FRAMES PROCESSED SUCCESSFULLY!")
print("================================")
print("Results saved in:", output_folder)