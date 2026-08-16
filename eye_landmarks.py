import cv2
import mediapipe as mp

LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

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

            for idx in LEFT_EYE + RIGHT_EYE:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Eye Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
