import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

MOUTH = [61, 291, 0, 17, 78, 308]
IMG_SIZE = 64
padding = 20

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

def predict_mouth(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)[0][0]
    return "YAWNING" if pred > 0.5 else "NORMAL"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "Detecting..."

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mx = [int(face_landmarks.landmark[i].x * w) for i in MOUTH]
            my = [int(face_landmarks.landmark[i].y * h) for i in MOUTH]

            x1 = max(min(mx) - padding, 0)
            x2 = min(max(mx) + padding, w)
            y1 = max(min(my) - padding, 0)
            y2 = min(max(my) + padding, h)

            mouth = frame[y1:y2, x1:x2]

            if mouth.size != 0:
                status = predict_mouth(mouth)

    color = (0,0,255) if status=="YAWNING" else (0,255,0)

    cv2.putText(frame, status, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Yawning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
