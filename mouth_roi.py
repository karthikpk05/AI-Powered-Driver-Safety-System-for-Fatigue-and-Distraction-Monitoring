import cv2
import mediapipe as mp

MOUTH = [61, 291, 81, 178, 13, 14]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            mouth_x = [int(face_landmarks.landmark[i].x * w) for i in MOUTH]
            mouth_y = [int(face_landmarks.landmark[i].y * h) for i in MOUTH]

            x1, x2 = min(mouth_x), max(mouth_x)
            y1, y2 = min(mouth_y), max(mouth_y)

            mouth_roi = frame[y1:y2, x1:x2]

            if mouth_roi.size != 0:
                cv2.imshow("Mouth ROI", mouth_roi)

            # Draw mouth landmarks
            for idx in MOUTH:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Mouth ROI Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
