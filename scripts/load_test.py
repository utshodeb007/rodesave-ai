import cv2
import os

# ==========================================
# Find Project Root Automatically
# ==========================================
script_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_folder)

# ==========================================
# Dataset Folder
# ==========================================
dataset_folder = os.path.join(
    project_root,
    "Data",
    "accident detection"
)

print("Project root:")
print(project_root)

print("\nDataset folder:")
print(dataset_folder)

# ==========================================
# Accident / Non-Accident Folders
# ==========================================
folders = {
    "accident": os.path.join(
        dataset_folder,
        "accident"
    ),

    "non_accident": os.path.join(
        dataset_folder,
        "non_accident"
    )
}

# ==========================================
# Check Folders
# ==========================================
for label, folder_path in folders.items():

    print("\n================================")
    print("Checking:", label)
    print("================================")

    print("Folder path:")
    print(folder_path)

    # Check folder exists
    if not os.path.isdir(folder_path):
        print("Folder not found!")
        continue

    # Find images
    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print("Images found:", len(images))

    if len(images) == 0:
        print("No images found!")
        continue

    # ======================================
    # Load First Image
    # ======================================
    image_path = os.path.join(
        folder_path,
        images[0]
    )

    print("Opening:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image!")
        continue

    print("Image loaded successfully!")
    print("File:", images[0])
    print("Image size:", image.shape)

    # Show image
    cv2.imshow(label, image)

# ==========================================
# Finish
# ==========================================
print("\n================================")
print("Dataset loading test completed!")
print("Press Q to close.")
print("================================")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

print("Test finished successfully!")