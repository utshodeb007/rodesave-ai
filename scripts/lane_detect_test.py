import cv2
import os
import numpy as np

# ==========================================
# 1. Frames Folder
# ==========================================
frames_folder = "data/frames"

if not os.path.exists(frames_folder):
    print("ERROR: Frames folder not found!")
    exit()

folders = [
    folder for folder in os.listdir(frames_folder)
    if not folder.startswith(".")
]

if len(folders) == 0:
    print("No frame folders found!")
    exit()

first_folder = folders[0]
folder_path = os.path.join(frames_folder, first_folder)

images = [
    image for image in os.listdir(folder_path)
    if image.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(images) == 0:
    print("No frames found!")
    exit()

# ==========================================
# 2. Load Frame
# ==========================================
image_path = os.path.join(folder_path, images[0])

print("Opening:", image_path)

frame = cv2.imread(image_path)

if frame is None:
    print("ERROR: Could not load frame!")
    exit()

print("Frame loaded successfully!")

# ==========================================
# 3. Resize
# ==========================================
frame = cv2.resize(frame, (1280, 720))

# ==========================================
# 4. Grayscale
# ==========================================
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("Grayscale conversion successful!")

# ==========================================
# 5. Gaussian Blur
# ==========================================
blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

print("Gaussian Blur successful!")

# ==========================================
# 6. Canny Edge Detection
# ==========================================
edges = cv2.Canny(
    blur,
    50,
    150
)

print("Canny Edge Detection successful!")

# ==========================================
# 7. Region of Interest
# ==========================================
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([
    [
        (0, height),
        (width, height),
        (int(width * 0.68), int(height * 0.50)),
        (int(width * 0.32), int(height * 0.50))
    ]
], dtype=np.int32)

cv2.fillPoly(
    mask,
    polygon,
    255
)

roi_edges = cv2.bitwise_and(
    edges,
    mask
)

print("Region of Interest applied!")

# ==========================================
# 8. Hough Line Detection
# ==========================================
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,
    minLineLength=40,
    maxLineGap=120
)

# Copy original frame
lane_frame = frame.copy()

# Lists for left and right lanes
left_lines = []
right_lines = []

# ==========================================
# 9. Separate Left and Right Lines
# ==========================================
if lines is not None:

    print("Hough lines detected:", len(lines))

    lines = np.asarray(lines).reshape(-1, 4)

    for line in lines:

        x1, y1, x2, y2 = line

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Avoid division by zero
        if x2 == x1:
            continue

        # Calculate slope
        slope = (y2 - y1) / (x2 - x1)

        # Ignore almost horizontal lines
        if abs(slope) < 0.5:
            continue

        # ======================================
        # Left Lane
        # ======================================
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))

        # ======================================
        # Right Lane
        # ======================================
        elif slope > 0:
            right_lines.append((x1, y1, x2, y2))

else:

    print("No Hough lines detected!")

# ==========================================
# 10. Function for Smooth Lane Line
# ==========================================
def make_smooth_line(line_list, height):

    if len(line_list) == 0:
        return None

    slopes = []
    intercepts = []

    for x1, y1, x2, y2 in line_list:

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    if len(slopes) == 0:
        return None

    # Average slope
    avg_slope = np.mean(slopes)

    # Average intercept
    avg_intercept = np.mean(intercepts)

    # Bottom point
    y_bottom = height

    # Upper point
    y_top = int(height * 0.55)

    # Calculate X positions
    x_bottom = int(
        (y_bottom - avg_intercept) / avg_slope
    )

    x_top = int(
        (y_top - avg_intercept) / avg_slope
    )

    return (
        x_bottom,
        y_bottom,
        x_top,
        y_top
    )


# ==========================================
# 11. Create Smooth Left Lane
# ==========================================
left_smooth = make_smooth_line(
    left_lines,
    height
)

# ==========================================
# 12. Create Smooth Right Lane
# ==========================================
right_smooth = make_smooth_line(
    right_lines,
    height
)

# ==========================================
# 13. Draw Smooth Left Lane
# ==========================================
if left_smooth is not None:

    x1, y1, x2, y2 = left_smooth

    cv2.line(
        lane_frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        6
    )

    print("Smooth LEFT lane detected!")

# ==========================================
# 14. Draw Smooth Right Lane
# ==========================================
if right_smooth is not None:

    x1, y1, x2, y2 = right_smooth

    cv2.line(
        lane_frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        6
    )

    print("Smooth RIGHT lane detected!")

# ==========================================
# 15. Draw ROI Boundary
# ==========================================
cv2.polylines(
    lane_frame,
    polygon,
    True,
    (255, 0, 0),
    2
)

# ==========================================
# 16. Display Results
# ==========================================
cv2.imshow(
    "Original Frame",
    frame
)

cv2.imshow(
    "Canny + ROI",
    roi_edges
)

cv2.imshow(
    "Smooth Lane Detection",
    lane_frame
)

print("----------------------------------")
print("Lane smoothing completed!")
print("Press Q to close.")
print("----------------------------------")

# ==========================================
# 17. Wait for Q
# ==========================================
while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==========================================
# 18. Close
# ==========================================
cv2.destroyAllWindows()

print("Test finished successfully!")