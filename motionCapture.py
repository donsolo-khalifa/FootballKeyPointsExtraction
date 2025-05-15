import cv2
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

# 3) Process all frames with tqdm
for _ in tqdm(range(total_frames), desc="Processing frames"):
    success, img = cap.read()
    if not success:
        break

    # Pose
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=False)

    # Ball
    ball_center = None
    for r in model(img, stream=False, verbose=False):
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                ball_center = (cx, cy)

    # Record
    if lmList:
        parts = []
        for lm in lmList:
            x_s = lm[0] / 100
            y_s = (img.shape[0] - lm[1]) / 100
            z_s = lm[2] / 300
            parts += [f"{x_s:.4f}", f"{y_s:.4f}", f"{z_s:.4f}"]

        # ball as 34th joint
        if ball_center:
            bx, by = ball_center
            parts += [f"{bx/100:.4f}", f"{(img.shape[0]-by)/100:.4f}", "0.0000"]
        else:
            # if ball not detected ball point returns to origin (0,0,0)
            parts += ["0.0000", "0.0000", "0.0000"]

        posList.append(",".join(parts))

cap.release()

# 4) Write out once
with open("AnimationFile.txt", "w") as f:
    f.write("\n".join(posList))

print("✅ Done — wrote", len(posList), "frames to AnimationFile.txt")
