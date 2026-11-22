import cv2
import os

# ==========================================
# Project paths
# ==========================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

dataset_folder = os.path.join(
    project_root,
    "Data",
    "accident detection"
)

# Input folders
accident_folder = os.path.join(
    dataset_folder,
    "accident"
)

non_accident_folder = os.path.join(
    dataset_folder,
    "non_accident"
)

# Output folder
frames_folder = os.path.join(
    dataset_folder,
    "frames"
)

accident_frames = os.path.join(
    frames_folder,
    "accident"
)

non_accident_frames = os.path.join(
    frames_folder,
    "non_accident"
)

# Create output folders
os.makedirs(accident_frames, exist_ok=True)
os.makedirs(non_accident_frames, exist_ok=True)


# ==========================================
# Function: Extract Frames
# ==========================================

def extract_frames(input_folder, output_folder, label):

    print("\n================================")
    print("Processing:", label)
    print("================================")

    if not os.path.exists(input_folder):
        print("Folder not found:", input_folder)
        return

    videos = [
        file for file in os.listdir(input_folder)
        if file.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    ]

    print("Videos found:", len(videos))

    if len(videos) == 0:
        print("No videos found!")
        return

    total_frames = 0

    # ======================================
    # Process each video
    # ======================================

    for video_number, video_file in enumerate(videos, start=1):

        video_path = os.path.join(
            input_folder,
            video_file
        )

        print(
            f"\nProcessing video {video_number}/{len(videos)}:"
        )

        print(video_file)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not open video!")
            continue

        frame_number = 0
        saved_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Save every 30th frame
            if frame_number % 30 == 0:

                output_name = (
                    f"{label}_{video_number}_"
                    f"frame_{saved_number:06d}.jpg"
                )

                output_path = os.path.join(
                    output_folder,
                    output_name
                )

                cv2.imwrite(
                    output_path,
                    frame
                )

                saved_number += 1
                total_frames += 1

        cap.release()

        print(
            "Frames saved from this video:",
            saved_number
        )

    print("\n--------------------------------")
    print(label, "completed!")
    print("Total frames saved:", total_frames)
    print("Output folder:", output_folder)
    print("--------------------------------")


# ==========================================
# Extract Accident Frames
# ==========================================

extract_frames(
    accident_folder,
    accident_frames,
    "accident"
)


# ==========================================
# Extract Non-Accident Frames
# ==========================================

extract_frames(
    non_accident_folder,
    non_accident_frames,
    "non_accident"
)


# ==========================================
# Final Message
# ==========================================

print("\n========================================")
print("DATASET FRAME EXTRACTION COMPLETED!")
print("========================================")

print("\nAccident frames:")
print(accident_frames)

print("\nNon-Accident frames:")
print(non_accident_frames)