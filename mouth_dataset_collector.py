import cv2
import mediapipe as mp
import os

# Mouth landmark indices
MOUTH = [61, 291, 0, 17, 78, 308]

save_yawn = "dataset/mouth/yawn"
save_normal = "dataset/mouth/normal"

os.makedirs(save_yawn, exist_ok=True)
os.makedirs(save_normal, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

padding = 20
count_yawn = 184
count_normal = 278

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    mouth_roi = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mx = [int(face_landmarks.landmark[i].x * w) for i in MOUTH]
            my = [int(face_landmarks.landmark[i].y * h) for i in MOUTH]

            x1 = max(min(mx) - padding, 0)
            x2 = min(max(mx) + padding, w)
            y1 = max(min(my) - padding, 0)
            y2 = min(max(my) + padding, h)

            mouth_roi = frame[y1:y2, x1:x2]

            if mouth_roi.size != 0:
                mouth_roi = cv2.resize(mouth_roi, (64, 64))
                cv2.imshow("Mouth ROI", mouth_roi)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('y') and mouth_roi is not None:
        cv2.imwrite(f"{save_yawn}/yawn_{count_yawn}.jpg", mouth_roi)
        count_yawn += 1
        print("Saved YAWN")

    elif key == ord('n') and mouth_roi is not None:
        cv2.imwrite(f"{save_normal}/normal_{count_normal}.jpg", mouth_roi)
        count_normal += 1
        print("Saved NORMAL")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
