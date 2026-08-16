import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained CNN model
model = tf.keras.models.load_model("models/eye_state_cnn.h5")

# Eye landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

padding = 16
IMG_SIZE = 64

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

def predict_eye_state(eye_img):
    eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye_img = eye_img / 255.0
    eye_img = np.expand_dims(eye_img, axis=0)
    pred = model.predict(eye_img, verbose=0)[0][0]
    return "OPEN" if pred > 0.5 else "CLOSED"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    state_text = "Detecting..."
    color = (255, 255, 0)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # LEFT EYE
            lx = [int(face_landmarks.landmark[i].x * w) for i in LEFT_EYE]
            ly = [int(face_landmarks.landmark[i].y * h) for i in LEFT_EYE]

            lx1 = max(min(lx) - padding, 0)
            lx2 = min(max(lx) + padding, w)
            ly1 = max(min(ly) - padding, 0)
            ly2 = min(max(ly) + padding, h)

            left_eye = frame[ly1:ly2, lx1:lx2]

            # RIGHT EYE
            rx = [int(face_landmarks.landmark[i].x * w) for i in RIGHT_EYE]
            ry = [int(face_landmarks.landmark[i].y * h) for i in RIGHT_EYE]

            rx1 = max(min(rx) - padding, 0)
            rx2 = min(max(rx) + padding, w)
            ry1 = max(min(ry) - padding, 0)
            ry2 = min(max(ry) + padding, h)

            right_eye = frame[ry1:ry2, rx1:rx2]

            if left_eye.size != 0 and right_eye.size != 0:
                left_state = predict_eye_state(left_eye)
                right_state = predict_eye_state(right_eye)

                if left_state == "CLOSED" and right_state == "CLOSED":
                    state_text = "EYES CLOSED"
                    color = (0, 0, 255)
                else:
                    state_text = "EYES OPEN"
                    color = (0, 255, 0)

    cv2.putText(frame, state_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Live Eye State Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
