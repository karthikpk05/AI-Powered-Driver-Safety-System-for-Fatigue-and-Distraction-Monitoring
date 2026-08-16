import cv2
import mediapipe as mp
import os

# MediaPipe eye landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

# Dataset paths
save_open = "dataset/eye/open"
save_closed = "dataset/eye/closed"

os.makedirs(save_open, exist_ok=True)
os.makedirs(save_closed, exist_ok=True)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Camera
cap = cv2.VideoCapture(0)

count_open = 377
count_closed = 403
padding = 16 # recommended padding (10–15)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    left_eye = None
    right_eye = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # ---------- LEFT EYE ----------
            left_x = [int(face_landmarks.landmark[i].x * w) for i in LEFT_EYE]
            left_y = [int(face_landmarks.landmark[i].y * h) for i in LEFT_EYE]

            lx1 = max(min(left_x) - padding, 0)
            lx2 = min(max(left_x) + padding, w)
            ly1 = max(min(left_y) - padding, 0)
            ly2 = min(max(left_y) + padding, h)

            left_eye = frame[ly1:ly2, lx1:lx2]

            if left_eye.size != 0:
                left_eye = cv2.resize(left_eye, (64, 64))
                cv2.imshow("Left Eye ROI", left_eye)

            # ---------- RIGHT EYE ----------
            right_x = [int(face_landmarks.landmark[i].x * w) for i in RIGHT_EYE]
            right_y = [int(face_landmarks.landmark[i].y * h) for i in RIGHT_EYE]

            rx1 = max(min(right_x) - padding, 0)
            rx2 = min(max(right_x) + padding, w)
            ry1 = max(min(right_y) - padding, 0)
            ry2 = min(max(right_y) + padding, h)

            right_eye = frame[ry1:ry2, rx1:rx2]

            if right_eye.size != 0:
                right_eye = cv2.resize(right_eye, (64, 64))
                cv2.imshow("Right Eye ROI", right_eye)

    key = cv2.waitKey(1) & 0xFF

    # ---------- SAVE OPEN EYES ----------
    if key == ord('o'):
        if left_eye is not None and left_eye.size != 0:
            cv2.imwrite(f"{save_open}/open_{count_open}_L.jpg", left_eye)
        if right_eye is not None and right_eye.size != 0:
            cv2.imwrite(f"{save_open}/open_{count_open}_R.jpg", right_eye)

        count_open += 1
        print("Saved OPEN eyes")

    # ---------- SAVE CLOSED EYES ----------
    elif key == ord('c'):
        if left_eye is not None and left_eye.size != 0:
            cv2.imwrite(f"{save_closed}/closed_{count_closed}_L.jpg", left_eye)
        if right_eye is not None and right_eye.size != 0:
            cv2.imwrite(f"{save_closed}/closed_{count_closed}_R.jpg", right_eye)

        count_closed += 1
        print("Saved CLOSED eyes")

    # ---------- EXIT ----------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
