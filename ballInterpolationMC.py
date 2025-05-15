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

posList = []
ball_positions = []  # Store ball positions for interpolation
frame_indices = []  # Store frame indices where ball was detected

# 3) First pass - collect all ball detections
for frame_idx in tqdm(range(total_frames), desc="Detecting ball positions"):
    success, img = cap.read()
    if not success:
        break

    # Ball detection
    ball_center = None
    for r in model(img, stream=False, verbose=False):
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                ball_center = (cx, cy)
                ball_positions.append(ball_center)
                frame_indices.append(frame_idx)
                break

    # Reset video position to beginning
    if frame_idx == total_frames - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 4) Interpolate missing ball positions
all_ball_positions = []
if len(ball_positions) > 1:  # Only interpolate if we have at least 2 detections
    for frame_idx in range(total_frames):
        if frame_idx in frame_indices:
            # Use actual detection
            ball_pos = ball_positions[frame_indices.index(frame_idx)]
            all_ball_positions.append(ball_pos)
        else:
            # Find nearest detections before and after current frame
            prev_idx = -1
            next_idx = -1

            for i, idx in enumerate(frame_indices):
                if idx < frame_idx:
                    prev_idx = i
                if idx > frame_idx and next_idx == -1:
                    next_idx = i
                    break

            # Interpolate
            if prev_idx != -1 and next_idx != -1:
                # Linear interpolation between two known positions
                prev_frame = frame_indices[prev_idx]
                next_frame = frame_indices[next_idx]
                prev_pos = ball_positions[prev_idx]
                next_pos = ball_positions[next_idx]

                # Calculate interpolation factor
                alpha = (frame_idx - prev_frame) / (next_frame - prev_frame)

                # Interpolate x and y
                x = int(prev_pos[0] + alpha * (next_pos[0] - prev_pos[0]))
                y = int(prev_pos[1] + alpha * (next_pos[1] - prev_pos[1]))

                all_ball_positions.append((x, y))
            elif prev_idx != -1:
                # If only have previous detections, use the last known position
                all_ball_positions.append(ball_positions[prev_idx])
            elif next_idx != -1:
                # If only have future detections, use the first known position
                all_ball_positions.append(ball_positions[next_idx])
            else:
                # No detections at all (shouldn't happen)
                all_ball_positions.append((0, 0))
else:
    # If fewer than 2 detections, can't interpolate
    all_ball_positions = [(0, 0)] * total_frames

# 5) Second pass - process all frames with interpolated ball positions
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
    success, img = cap.read()
    if not success:
        break

    # Pose detection
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=False)

    # Get interpolated ball position
    ball_center = all_ball_positions[frame_idx] if frame_idx < len(all_ball_positions) else (0, 0)

    # Record
    if lmList:
        parts = []
        for lm in lmList:
            x_s = lm[0] / 100
            y_s = (img.shape[0] - lm[1]) / 100
            z_s = lm[2] / 300
            parts += [f"{x_s:.4f}", f"{y_s:.4f}", f"{z_s:.4f}"]

        # ball as 34th joint (now interpolated when needed)
        bx, by = ball_center
        parts += [f"{bx / 100:.4f}", f"{(img.shape[0] - by) / 100:.4f}", "0.0000"]

        posList.append(",".join(parts))

cap.release()

# 6) Write out once
with open("AnimationFile.txt", "w") as f:
    f.write("\n".join(posList))

print("✅ Done — wrote", len(posList), "frames to AnimationFile.txt")
print(
    f"Ball was detected in {len(frame_indices)} frames and interpolated in {total_frames - len(frame_indices)} frames")