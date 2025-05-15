import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
from tqdm import tqdm

# 1) Open video & get total frames
video_path = 'vida.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 2) Detectors
detector = PoseDetector()
model = YOLO('../QuadraticParabola/yolo-weights/yolov8x.pt')
BALL_CLASS_ID = 32

# Store detected ball positions and their frame indices
detected_ball_positions = []
detected_frame_indices = []

# 3) First pass - collect all ball detections
for frame_idx in tqdm(range(total_frames), desc="Detecting ball positions"):
    success, img = cap.read()
    if not success:
        break

    # Ball detection
    for r in model(img, stream=False, verbose=False):
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                detected_ball_positions.append((cx, cy))
                detected_frame_indices.append(frame_idx)
                break # Assuming only one ball per frame

# Convert to numpy arrays for easier handling
detected_frame_indices = np.array(detected_frame_indices)
detected_ball_positions = np.array(detected_ball_positions)

# 4) Interpolate missing ball positions using NumPy
all_ball_positions = np.zeros((total_frames, 2), dtype=np.int32) # Initialize array for all frames

if len(detected_frame_indices) > 1:
    # Create an array of all frame indices
    all_frame_indices = np.arange(total_frames)

    # Use numpy.interp for linear interpolation
    # It expects x-coordinates (frame indices where data is known),
    # fp-coordinates (the known data points, ball positions),
    # and xp-coordinates (the new x-coordinates to interpolate at, all frame indices)

    # Interpolate X coordinates
    all_ball_positions[:, 0] = np.interp(
        all_frame_indices,          # The x-coordinates where we want to interpolate (all frame indices)
        detected_frame_indices,     # The x-coordinates of the data points (detected frame indices)
        detected_ball_positions[:, 0] # The y-coordinates of the data points (detected ball X positions)
    )

    # Interpolate Y coordinates
    all_ball_positions[:, 1] = np.interp(
        all_frame_indices,          # The x-coordinates where we want to interpolate (all frame indices)
        detected_frame_indices,     # The x-coordinates of the data points (detected frame indices)
        detected_ball_positions[:, 1] # The y-coordinates of the data points (detected ball Y positions)
    )

elif len(detected_frame_indices) == 1:
     # If only one detection, fill all frames with that position
     all_ball_positions[:, 0] = detected_ball_positions[0, 0]
     all_ball_positions[:, 1] = detected_ball_positions[0, 1]
else:
    # If no detections, all positions remain (0, 0) as initialized
    pass

# 5) Second pass - process all frames with interpolated ball positions
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
posList = [] # Reset posList for the second pass
for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
    success, img = cap.read()
    if not success:
        break

    # Pose detection
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=False)

    # Get interpolated ball position from the NumPy array
    ball_center = all_ball_positions[frame_idx]

    # Record
    if lmList:
        parts = []
        for lm in lmList:
            x_s = lm[0] / 100
            y_s = (img.shape[0] - lm[1]) / 100
            z_s = lm[2] / 300
            parts += [f"{x_s:.4f}", f"{y_s:.4f}", f"{z_s:.4f}"]

        # ball as 34th joint (now interpolated from NumPy array)
        bx, by = ball_center
        parts += [f"{bx / 100:.4f}", f"{(img.shape[0] - by) / 100:.4f}", "0.0000"]

        posList.append(",".join(parts))

cap.release()

# 6) Write out once
with open("AnimationFile.txt", "w") as f:
    f.write("\n".join(posList))

print("✅ Done — wrote", len(posList), "frames to AnimationFile.txt")
print(
    f"Ball was detected in {len(detected_frame_indices)} frames and interpolated in {total_frames - len(detected_frame_indices)} frames")
