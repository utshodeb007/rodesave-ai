import os
import cv2
import numpy as np
import random

# ==========================================
# Project Root
# ==========================================

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

project_root = os.path.dirname(script_folder)

# ==========================================
# Dataset Paths
# ==========================================

base_folder = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "frames"
)

accident_folder = os.path.join(
    base_folder,
    "accident"
)

non_accident_folder = os.path.join(
    base_folder,
    "non_accident"
)

# Output dataset folder
output_folder = os.path.join(
    project_root,
    "Data",
    "accident detection",
    "prepared_dataset"
)

train_folder = os.path.join(
    output_folder,
    "train"
)

test_folder = os.path.join(
    output_folder,
    "test"
)

# Create folders
os.makedirs(
    os.path.join(train_folder, "accident"),
    exist_ok=True
)

os.makedirs(
    os.path.join(train_folder, "non_accident"),
    exist_ok=True
)

os.makedirs(
    os.path.join(test_folder, "accident"),
    exist_ok=True
)

os.makedirs(
    os.path.join(test_folder, "non_accident"),
    exist_ok=True
)

# ==========================================
# Settings
# ==========================================

IMAGE_SIZE = (128, 128)
TRAIN_RATIO = 0.80

# ==========================================
# Prepare Images
# ==========================================

def prepare_class(
    input_folder,
    class_name
):

    print("\n================================")
    print("Processing:", class_name)
    print("================================")

    if not os.path.exists(input_folder):
        print("Folder not found:")
        print(input_folder)
        return

    images = [
        file for file in os.listdir(input_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print("Images found:", len(images))

    if len(images) == 0:
        return

    # Shuffle images
    random.shuffle(images)

    # Calculate training images
    train_count = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:train_count]
    test_images = images[train_count:]

    print("Training images:", len(train_images))
    print("Testing images:", len(test_images))

    # ======================================
    # Training Images
    # ======================================

    for index, image_name in enumerate(
        train_images
    ):

        image_path = os.path.join(
            input_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        output_path = os.path.join(
            train_folder,
            class_name,
            f"{class_name}_train_{index:06d}.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

    # ======================================
    # Testing Images
    # ======================================

    for index, image_name in enumerate(
        test_images
    ):

        image_path = os.path.join(
            input_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        output_path = os.path.join(
            test_folder,
            class_name,
            f"{class_name}_test_{index:06d}.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

    print(
        class_name,
        "completed successfully!"
    )


# ==========================================
# Accident Dataset
# ==========================================

prepare_class(
    accident_folder,
    "accident"
)

# ==========================================
# Non-Accident Dataset
# ==========================================

prepare_class(
    non_accident_folder,
    "non_accident"
)

# ==========================================
# Final Message
# ==========================================

print("\n========================================")
print("DATASET PREPARATION COMPLETED!")
print("========================================")

print("\nPrepared dataset:")
print(output_folder)

print("\nTraining folder:")
print(train_folder)

print("\nTesting folder:")
print(test_folder)