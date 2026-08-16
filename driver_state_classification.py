#driver_state_classification.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from ultralytics import YOLO
from voice_alert import driver_alert

# ================= LOAD MODELS =================

eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

yolo_model = YOLO("yolov8n.pt")

LEFT_EYE = [33,133,160,159,158,144]
MOUTH = [61,291,0,17,78,308]

IMG = 64
padding = 16

EYE_TIME_THRESHOLD = 2.0
MOUTH_FRAMES = 8

OBSERVE_TIME = 20
ALERT_HOLD = 8

eye_close_start = None
eye_event_triggered = False

long_blinks = 0
yawn_count = 0

window_start = None
alert_start = None
state = "AWAKE"

mouth_open_frames = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)

cap = cv2.VideoCapture(0)

def preprocess(img):
    img=cv2.resize(img,(IMG,IMG))
    img=img/255.0
    return np.expand_dims(img,0)

while True:

    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    eye_closed=False
    yawning=False

    now=time.time()

    # ================= MODULE 6 (UNCHANGED) =================

    if results.multi_face_landmarks:

        lm=results.multi_face_landmarks[0]

        ex=[int(lm.landmark[i].x*w) for i in LEFT_EYE]
        ey=[int(lm.landmark[i].y*h) for i in LEFT_EYE]
        eye=frame[min(ey)-padding:max(ey)+padding, min(ex)-padding:max(ex)+padding]

        mx=[int(lm.landmark[i].x*w) for i in MOUTH]
        my=[int(lm.landmark[i].y*h) for i in MOUTH]
        mouth=frame[min(my)-padding:max(my)+padding, min(mx)-padding:max(mx)+padding]

        if eye.size:
            p=eye_model.predict(preprocess(eye),verbose=0)[0][0]
            eye_closed = p<0.5

        if mouth.size:
            m=mouth_model.predict(preprocess(mouth),verbose=0)[0][0]
            yawning = m<0.5

    # ---------- EYE TEMPORAL ----------

    if eye_closed:

        if eye_close_start is None:
            eye_close_start = now

        if now-eye_close_start>=EYE_TIME_THRESHOLD and not eye_event_triggered:
            long_blinks+=1
            eye_event_triggered=True
            print("Long eye closure")

    else:
        eye_close_start=None
        eye_event_triggered=False

    # ---------- YAWN ----------

    if yawning:
        mouth_open_frames+=1
    else:
        if mouth_open_frames>=MOUTH_FRAMES:
            yawn_count+=1
            print("Yawn")
        mouth_open_frames=0

    # ---------- START WINDOW ----------

    if window_start is None and (long_blinks>0 or yawn_count>0):
        window_start=now

    # ---------- DECISION ----------

    if window_start:

        if now-window_start>OBSERVE_TIME:
            long_blinks=0
            yawn_count=0
            window_start=None
            state="AWAKE"

        if long_blinks>=1 or yawn_count>=2:
            state="TIRED"
            alert_start=now

        if long_blinks>=1 and yawn_count>=2:
            state="DROWSY"
            alert_start=now

        if long_blinks>=3:
            state="CRITICAL"
            alert_start=now

    # ---------- ALERT HOLD ----------

    if alert_start and now-alert_start>ALERT_HOLD:
        alert_start=None
        window_start=None
        long_blinks=0
        yawn_count=0
        state="AWAKE"

    # ================= MODULE 7 (PHONE DETECTION) =================

    phone_detected = False

    results_yolo = yolo_model(frame)

    for r in results_yolo:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label == "cell phone":

                phone_detected = True

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,"PHONE",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    # ================= DISPLAY =================

    color=(0,255,0)
    if state=="TIRED": color=(0,255,255)
    if state=="DROWSY": color=(0,165,255)
    if state=="CRITICAL": color=(0,0,255)

    cv2.putText(frame,state,(30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.putText(frame,f"Blink:{long_blinks}",(30,80),0,0.7,(255,255,255),2)
    cv2.putText(frame,f"Yawn:{yawn_count}",(30,110),0,0.7,(255,255,255),2)

    # ---------- MOBILE WARNING ----------

    if phone_detected:
        cv2.putText(frame,"DON'T USE MOBILE PHONE",(30,150),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        
    driver_alert(state, phone_detected)

    cv2.imshow("Driver Monitoring System",frame)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()