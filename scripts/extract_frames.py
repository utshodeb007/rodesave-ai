import cv2
import os

# Input video folder
video_folder = "data/raw_videos"

# Output folder for extracted frames
output_folder = "data/frames"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get all files from raw_videos
videos = os.listdir(video_folder)

# Process only video files
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

videos = [
    video for video in videos
    if video.lower().endswith(video_extensions)
]

print("Videos found:", len(videos))

# Process every video
for video in videos:

    video_path = os.path.join(video_folder, video)

    # Create a separate folder for each video's frames
    video_name = os.path.splitext(video)[0]
    video_output_folder = os.path.join(output_folder, video_name)

    os.makedirs(video_output_folder, exist_ok=True)

    print("\nProcessing:", video)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open:", video)
        continue

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Save every 30th frame
        if frame_count % 30 == 0:

            frame_name = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(
                video_output_folder,
                frame_name
            )

            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

    print("Total frames:", frame_count)
    print("Frames saved in:", video_output_folder)

print("\nAll videos processed successfully!")