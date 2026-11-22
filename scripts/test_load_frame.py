import cv2
import os

# Frames folder
frames_folder = "data/frames"

# Check if frames folder exists
if not os.path.exists(frames_folder):
    print("ERROR: Frames folder not found!")
    exit()

# Find video/frame folders
folders = [
    folder for folder in os.listdir(frames_folder)
    if not folder.startswith(".")
]

print("Frame folders found:", len(folders))

if len(folders) == 0:
    print("No frame folder found!")
    exit()

# Select the first video's frame folder
first_folder = folders[0]
folder_path = os.path.join(frames_folder, first_folder)

# Find image files
images = [
    image for image in os.listdir(folder_path)
    if image.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Images found:", len(images))

if len(images) == 0:
    print("No image found!")
    exit()

# Load first frame
image_path = os.path.join(folder_path, images[0])

print("Opening:", image_path)

frame = cv2.imread(image_path)

if frame is not None:
    print("Frame loaded successfully!")

    cv2.imshow("RoadSafe AI - Frame Test", frame)

    print("Press Q to close.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

else:
    print("ERROR: Could not load frame!")