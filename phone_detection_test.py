import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                phone_detected = True

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,"PHONE DETECTED",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    if phone_detected:
        cv2.putText(frame,"DRIVER DISTRACTED",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    else:
        cv2.putText(frame,"NO PHONE",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.imshow("Phone Detection",frame)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()