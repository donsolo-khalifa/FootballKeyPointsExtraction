import cv2
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import cvzone


cap = cv2.VideoCapture('vida.mp4')

detector = PoseDetector()
posList = []

model = YOLO('yolo11x.pt')
names = model.names
BALL_CLASS_ID = 32  # COCO index for sports ball 32 for yolov8 coco


# target display size (so the window never exceeds this)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# create a resizable window and set its display size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", DISPLAY_WIDTH, DISPLAY_HEIGHT)

while True:
    success, img = cap.read()
    results = model(img, stream=False, verbose=False)

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                best_conf = conf
                ball_center = (cx, cy)
                # draw detection
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 255, 0))
                cvzone.putTextRect(img, f"ball {conf:.2f}", (x1, y1 - 10),
                                   colorR=(255, 255, 0), scale=1, thickness=1)

    # if bboxInfo:
    #     lmString = ''
    #     for lm in lmList:
    #         lmString += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
    #         print(lm[2])
    #
    #     posList.append(lmString)


    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()