# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time

# # ================= LOAD MODELS =================

# eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
# mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

# LEFT_EYE = [33,133,160,159,158,144]
# MOUTH = [61,291,0,17,78,308]

# IMG = 64
# padding = 16

# # ---------- FRAME PARAMETERS ----------
# EYE_TIME_THRESHOLD = 2.0   # seconds

# eye_close_start = None
# eye_event_triggered = False

# EYE_FRAMES = 12      # ~2 seconds
# MOUTH_FRAMES = 8    # ~1 second

# OBSERVE_TIME = 20
# ALERT_HOLD = 8

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)

# cap = cv2.VideoCapture(0)

# eye_closed_frames = 0
# mouth_open_frames = 0

# long_blinks = 0
# yawn_count = 0

# window_start = None
# alert_start = None
# state = "AWAKE"

# def preprocess(img):
#     img=cv2.resize(img,(IMG,IMG))
#     img=img/255.0
#     return np.expand_dims(img,0)

# while True:

#     ret,frame=cap.read()
#     if not ret:
#         break

#     frame=cv2.flip(frame,1)
#     h,w,_=frame.shape

#     rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     results=face_mesh.process(rgb)

#     eye_closed=False
#     yawning=False

#     if results.multi_face_landmarks:

#         lm=results.multi_face_landmarks[0]

#         # -------- EYE ROI --------
#         ex=[int(lm.landmark[i].x*w) for i in LEFT_EYE]
#         ey=[int(lm.landmark[i].y*h) for i in LEFT_EYE]

#         e1,e2=min(ex)-padding,max(ex)+padding
#         f1,f2=min(ey)-padding,max(ey)+padding

#         eye=frame[f1:f2,e1:e2]

#         # -------- MOUTH ROI --------
#         mx=[int(lm.landmark[i].x*w) for i in MOUTH]
#         my=[int(lm.landmark[i].y*h) for i in MOUTH]

#         m1,m2=min(mx)-padding,max(mx)+padding
#         n1,n2=min(my)-padding,max(my)+padding

#         mouth=frame[n1:n2,m1:m2]

#         if eye.size:
#             p=eye_model.predict(preprocess(eye),verbose=0)[0][0]
#             eye_closed = p<0.5

#         if mouth.size:
#             m=mouth_model.predict(preprocess(mouth),verbose=0)[0][0]
#             yawning = m<0.5

#     # ---------- FRAME COUNTERS ----------

#     if eye_closed:
#         eye_closed_frames+=1
#     else:
#         if eye_closed_frames>=EYE_FRAMES:
#             long_blinks+=1
#         eye_closed_frames=0

#     if yawning:
#         mouth_open_frames+=1
#     else:
#         if mouth_open_frames>=MOUTH_FRAMES:
#             yawn_count+=1
#         mouth_open_frames=0

#     # ---------- START WINDOW ----------

#     now=time.time()

#     if window_start is None and (long_blinks>0 or yawn_count>0):
#         window_start=now

#     # ---------- DECISION ----------

#     if window_start:

#         if now-window_start>OBSERVE_TIME:
#             long_blinks=0
#             yawn_count=0
#             window_start=None
#             state="AWAKE"

#         if long_blinks>=1 or yawn_count>=2:
#             state="TIRED"
#             alert_start=now

#         if long_blinks>=1 and yawn_count>=2:
#             state="DROWSY"
#             alert_start=now

#         if long_blinks>=3:
#             state="CRITICAL"
#             alert_start=now

#     # ---------- ALERT HOLD ----------

#     if alert_start and now-alert_start>ALERT_HOLD:
#         alert_start=None
#         window_start=None
#         long_blinks=0
#         yawn_count=0
#         state="AWAKE"

#     # ---------- DISPLAY ----------

#     color=(0,255,0)
#     if state=="DROWSY": color=(0,165,255)
#     if state in ["CRITICAL","MICROSLEEP"]: color=(0,0,255)

#     cv2.putText(frame,state,(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
#     cv2.putText(frame,f"Blink:{long_blinks}",(30,80),0,0.7,(255,255,255),2)
#     cv2.putText(frame,f"Yawn:{yawn_count}",(30,110),0,0.7,(255,255,255),2)

#     cv2.imshow("Temporal Fatigue",frame)

#     if cv2.waitKey(1)==ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

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

    color=(0,255,0)
    if state=="TIRED": color=(0,255,255)
    if state=="DROWSY": color=(0,165,255)
    if state=="CRITICAL": color=(0,0,255)

    cv2.putText(frame,state,(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    cv2.putText(frame,f"Blink:{long_blinks}",(30,80),0,0.7,(255,255,255),2)
    cv2.putText(frame,f"Yawn:{yawn_count}",(30,110),0,0.7,(255,255,255),2)

    cv2.imshow("Temporal Fatigue",frame)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import time

# # ================= LOAD MODELS =================

# eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
# mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

# IMG = 64

# LEFT_EYE = [33,133,160,159,158,157,173,144,145,153]
# MOUTH = [61,185,40,39,37,0,267,269,270,409,291]

# # ================= PARAMETERS =================

# EYE_THRESH = 0.40
# MOUTH_THRESH = 0.35

# BLINK_CONFIRM = 4
# YAWN_CONFIRM = 6

# LONG_BLINK = 1.0
# OBSERVE_TIME = 20
# ALERT_HOLD = 8
# YAWN_COOLDOWN = 3

# # ================= STATE =================

# blink_start = None
# long_blinks = 0
# yawn_count = 0

# window_start = None
# alert_start = None
# state = "AWAKE"

# eye_closed_frames = 0
# mouth_open_frames = 0

# last_yawn_time = 0

# mp_face = mp.solutions.face_mesh
# mesh = mp_face.FaceMesh(refine_landmarks=True)

# cap = cv2.VideoCapture(0)

# # ================= HELPERS =================

# def preprocess(img):
#     img=cv2.resize(img,(IMG,IMG))
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img.astype("float32")/255.0
#     return np.expand_dims(img,0)

# def roi(frame,lm,idx):
#     h,w=frame.shape[:2]
#     pts=[[int(lm.landmark[i].x*w),int(lm.landmark[i].y*h)] for i in idx]
#     x,y,w1,h1=cv2.boundingRect(np.array(pts))
#     return frame[y:y+h1,x:x+w1]

# # ================= MAIN LOOP =================

# while True:

#     ret,frame=cap.read()
#     if not ret: break

#     frame=cv2.flip(frame,1)
#     rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     res=mesh.process(rgb)

#     now=time.time()
#     eye_closed=False
#     yawning=False

#     if res.multi_face_landmarks:

#         lm=res.multi_face_landmarks[0]

#         eye_img=roi(frame,lm,LEFT_EYE)
#         mouth_img=roi(frame,lm,MOUTH)

#         # -------- EYE --------
#         if eye_img.size:
#             p=eye_model.predict(preprocess(eye_img),verbose=0)[0][0]

#             if p<EYE_THRESH:
#                 eye_closed_frames+=1
#             else:
#                 eye_closed_frames=0

#         # -------- MOUTH --------
#         if mouth_img.size:
#             m=mouth_model.predict(preprocess(mouth_img),verbose=0)[0][0]

#             if m<MOUTH_THRESH:
#                 mouth_open_frames+=1
#             else:
#                 mouth_open_frames=0

#     eye_closed = eye_closed_frames>=BLINK_CONFIRM
#     yawning = mouth_open_frames>=YAWN_CONFIRM

#     # -------- START WINDOW --------

#     if window_start is None and (eye_closed or yawning):
#         window_start=now
#         long_blinks=0
#         yawn_count=0

#     # -------- BLINK LOGIC --------

#     if eye_closed and blink_start is None:
#         blink_start=now

#     if not eye_closed and blink_start:

#         dur=now-blink_start
#         blink_start=None

#         if dur>LONG_BLINK:
#             long_blinks+=1
#             print("Long blink")

#     # -------- YAWN LOGIC --------

#     if yawning and now-last_yawn_time>YAWN_COOLDOWN:
#         yawn_count+=1
#         last_yawn_time=now
#         mouth_open_frames=0
#         print("Yawn")

#     # -------- DECISION --------

#     if window_start:

#         if now-window_start>OBSERVE_TIME:
#             window_start=None
#             long_blinks=0
#             yawn_count=0
#             state="AWAKE"

#         if long_blinks>=1 and yawn_count>=1:
#             state="DROWSY"
#             alert_start=now

#         if long_blinks>=2 and yawn_count>=1:
#             state="CRITICAL"
#             alert_start=now

#         if long_blinks>=3:
#             state="MICROSLEEP"
#             alert_start=now

#     # -------- ALERT HOLD --------

#     if alert_start and now-alert_start>ALERT_HOLD:
#         alert_start=None
#         window_start=None
#         long_blinks=0
#         yawn_count=0
#         state="AWAKE"

#     # -------- DISPLAY --------

#     color=(0,255,0)
#     if state=="DROWSY": color=(0,165,255)
#     if state in ["CRITICAL","MICROSLEEP"]: color=(0,0,255)

#     cv2.putText(frame,state,(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
#     cv2.putText(frame,f"Blink:{long_blinks}",(30,80),0,0.7,(255,255,255),2)
#     cv2.putText(frame,f"Yawn:{yawn_count}",(30,110),0,0.7,(255,255,255),2)

#     cv2.imshow("Temporal Fatigue",frame)

#     if cv2.waitKey(1)==ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import time

# # ================= LOAD MODELS =================

# eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
# mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

# IMG = 64

# # Mediapipe landmarks
# LEFT_EYE = [33,133,160,159,158,157,173,144,145,153]
# MOUTH = [61,185,40,39,37,0,267,269,270,409,291]

# # ================= PARAMETERS =================

# EYE_THRESH = 0.35
# MOUTH_THRESH = 0.25

# LONG_BLINK = 1.2
# OBSERVE_TIME = 20
# ALERT_HOLD = 10
# YAWN_COOLDOWN = 2

# # ================= STATE =================

# blink_start = None
# long_blinks = 0
# yawn_count = 0

# window_start = None
# alert_start = None
# state = "AWAKE"
# last_yawn = 0

# eye_buffer = []
# mouth_buffer = []

# mp_face = mp.solutions.face_mesh
# mesh = mp_face.FaceMesh(refine_landmarks=True)

# cap = cv2.VideoCapture(0)

# # ================= HELPERS =================

# def preprocess(img):
#     img=cv2.resize(img,(IMG,IMG))
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img.astype("float32")/255.0
#     return np.expand_dims(img,0)

# def roi(frame,lm,idx):
#     h,w=frame.shape[:2]
#     pts=[]
#     for i in idx:
#         pts.append([int(lm.landmark[i].x*w),int(lm.landmark[i].y*h)])
#     x,y,w1,h1=cv2.boundingRect(np.array(pts))
#     return frame[y:y+h1,x:x+w1]

# # ================= MAIN LOOP =================

# while True:

#     ret,frame=cap.read()
#     if not ret:
#         break

#     frame=cv2.flip(frame,1)
#     rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     res=mesh.process(rgb)

#     eye_closed=False
#     yawning=False

#     now=time.time()

#     if res.multi_face_landmarks:

#         lm=res.multi_face_landmarks[0]

#         eye_img=roi(frame,lm,LEFT_EYE)
#         mouth_img=roi(frame,lm,MOUTH)

#         if eye_img.size:
#             p=eye_model.predict(preprocess(eye_img),verbose=0)[0][0]
#             eye_buffer.append(p)
#             if len(eye_buffer)>5: eye_buffer.pop(0)
#             eye_avg=sum(eye_buffer)/len(eye_buffer)
#             eye_closed=eye_avg<EYE_THRESH

#         if mouth_img.size:
#             m=mouth_model.predict(preprocess(mouth_img),verbose=0)[0][0]
#             mouth_buffer.append(m)
#             if len(mouth_buffer)>5: mouth_buffer.pop(0)
#             mouth_avg=sum(mouth_buffer)/len(mouth_buffer)
#             yawning=mouth_avg<MOUTH_THRESH

#     # ---------- START WINDOW ----------

#     if window_start is None and (eye_closed or yawning):
#         window_start=now
#         long_blinks=0
#         yawn_count=0

#     # ---------- BLINK LOGIC ----------

#     # if eye_closed:
#     #     if blink_start is None:
#     #         blink_start=now
#     # else:
#     #     if blink_start:
#     #         dur=now-blink_start
#     #         print("Blink:",dur)
#     #         if dur>LONG_BLINK:
#     #             long_blinks+=1
#     #             print("Long blink detected")
#     #         blink_start=None

#     # ---------- BLINK LOGIC (FIXED) ----------

#     if eye_closed and blink_start is None:
#         blink_start = now

#     if not eye_closed and blink_start is not None:

#         dur = now - blink_start
#         blink_start = None

#         print("Blink:", dur)

#         if dur > LONG_BLINK:
#             long_blinks += 1
#             print("Long blink detected")


#     # ---------- YAWN LOGIC ----------

#     if yawning and now-last_yawn>YAWN_COOLDOWN:
#         yawn_count+=1
#         last_yawn=now
#         print("Yawn detected")

#     # ---------- DECISION ----------

#     if window_start:

#         if now-window_start>OBSERVE_TIME:
#             window_start=None
#             long_blinks=0
#             yawn_count=0
#             state="AWAKE"

#         if long_blinks>=1 and yawn_count>=1:
#             state="DROWSY"
#             alert_start=now

#         if long_blinks>=2 and yawn_count>=1:
#             state="CRITICAL"
#             alert_start=now

#         if long_blinks>=3:
#             state="MICROSLEEP"
#             alert_start=now

#     # ---------- ALERT HOLD ----------

#     if alert_start and now-alert_start>ALERT_HOLD:
#         alert_start=None
#         window_start=None
#         long_blinks=0
#         yawn_count=0
#         state="AWAKE"

#     # ---------- DISPLAY ----------

#     color=(0,255,0)
#     if state=="DROWSY": color=(0,165,255)
#     if state=="CRITICAL": color=(0,0,255)
#     if state=="MICROSLEEP": color=(0,0,255)

#     cv2.putText(frame,state,(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
#     cv2.putText(frame,f"Blink:{long_blinks}",(30,80),0,0.7,(0,0,255),2)
#     cv2.putText(frame,f"Yawn:{yawn_count}",(30,110),0,0.7,(0,0,255),2)

#     cv2.imshow("Temporal Fatigue",frame)

#     if cv2.waitKey(1)==ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

