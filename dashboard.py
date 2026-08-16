# # dashboard.py  ──  Driver Monitoring System  ──  Module 10
# # Run:  streamlit run dashboard.py

# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import time
# import threading
# import queue
# import plotly.graph_objects as go
# from ultralytics import YOLO
# from voice_alert import driver_alert

# # ──────────────────────────────────────────────────────────────────────────────
# # PAGE CONFIG
# # ──────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Driver Monitoring System",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ──────────────────────────────────────────────────────────────────────────────
# # SESSION STATE INIT
# # ──────────────────────────────────────────────────────────────────────────────
# def _init_state():
#     defaults = dict(
#         frame_queue=None, _stop_event=None,
#         bg_thread=None, latest={},
#         session_start=time.time(),
#         dark_mode=False,
#         chart_type="bar",   # "bar" or "pie"
#     )
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# _init_state()
# ss = st.session_state

# # ──────────────────────────────────────────────────────────────────────────────
# # THEME VARIABLES
# # ──────────────────────────────────────────────────────────────────────────────
# if ss.dark_mode:
#     BG          = "#0F172A"
#     BG2         = "#1E293B"
#     CARD_BG     = "rgba(30,41,59,0.95)"
#     CARD_BORDER = "rgba(100,150,200,0.18)"
#     TEXT_PRI    = "#E2EAF4"
#     TEXT_SEC    = "#7A9AB8"
#     TEXT_MUT    = "#3A5570"
#     TOPBAR_BG   = "rgba(15,23,42,0.97)"
#     TOPBAR_BDR  = "rgba(56,189,248,0.2)"
#     ACCENT      = "#38BDF8"
#     PLOT_PAPER  = "rgba(0,0,0,0)"
#     PLOT_PLOT   = "rgba(0,0,0,0)"
#     GRID_COLOR  = "rgba(56,189,248,0.08)"
#     TICK_COLOR  = "#3A5570"
#     LABEL_COLOR = "#7A9AB8"
# else:
#     BG          = "#F0F4F8"
#     BG2         = "#FFFFFF"
#     CARD_BG     = "rgba(255,255,255,0.97)"
#     CARD_BORDER = "rgba(100,150,200,0.22)"
#     TEXT_PRI    = "#1A2940"
#     TEXT_SEC    = "#4A6A88"
#     TEXT_MUT    = "#8AAAC0"
#     TOPBAR_BG   = "rgba(255,255,255,0.97)"
#     TOPBAR_BDR  = "rgba(100,160,220,0.25)"
#     ACCENT      = "#0369A1"
#     PLOT_PAPER  = "rgba(255,255,255,0)"
#     PLOT_PLOT   = "rgba(255,255,255,0)"
#     GRID_COLOR  = "rgba(100,160,220,0.12)"
#     TICK_COLOR  = "#8AAAC0"
#     LABEL_COLOR = "#4A6A88"

# # ──────────────────────────────────────────────────────────────────────────────
# # GLOBAL CSS
# # ──────────────────────────────────────────────────────────────────────────────
# st.markdown(f"""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

# /* ── reset & viewport lock (no scroll) ── */
# html, body {{
#   overflow: hidden !important;
#   height: 100vh !important;
#   margin: 0; padding: 0;
# }}
# html, body, [data-testid="stAppViewContainer"] {{
#   background: {BG} !important;
#   color: {TEXT_PRI};
#   font-family: 'Lora', 'Times New Roman', Georgia, serif;
# }}
# [data-testid="stAppViewContainer"] {{
#   height: 100vh !important;
#   overflow: hidden !important;
# }}
# [data-testid="stHeader"], [data-testid="stToolbar"],
# footer, #MainMenu {{ display: none !important; }}
# .block-container {{
#   padding: 0 !important;
#   max-width: 100% !important;
#   height: 100vh !important;
#   overflow: hidden !important;
# }}
# section[data-testid="stSidebar"] {{ display: none !important; }}

# /* ── hide streamlit scrollbars ── */
# [data-testid="stVerticalBlock"] {{
#   overflow: hidden !important;
#   gap: 0 !important;
# }}

# /* ── top bar ── */
# .topbar {{
#   display: flex; align-items: center; gap: 16px;
#   padding: 10px 28px 9px;
#   border-bottom: 1px solid {TOPBAR_BDR};
#   background: {TOPBAR_BG};
#   backdrop-filter: blur(16px);
#   box-shadow: 0 1px 12px rgba(0,0,0,0.07);
# }}
# .topbar-logo {{
#   width: 38px; height: 38px; border-radius: 9px;
#   background: linear-gradient(135deg, #0EA5E9, #0369A1);
#   display: flex; align-items: center; justify-content: center;
#   font-size: 20px; flex-shrink: 0;
#   box-shadow: 0 2px 10px rgba(14,165,233,0.3);
# }}
# .topbar-title {{
#   font-family: 'IM Fell English', 'Times New Roman', serif;
#   font-size: 1.15rem; font-weight: 400; font-style: italic;
#   letter-spacing: 1px; color: {TEXT_PRI}; line-height: 1.1;
# }}
# .topbar-sub {{
#   font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
#   color: {TEXT_SEC}; letter-spacing: 2px; margin-top: 2px;
#   text-transform: uppercase;
# }}
# .topbar-right {{ margin-left: auto; display: flex; align-items: center; gap: 12px; }}
# .topbar-clock {{
#   font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
#   color: {ACCENT}; border: 1px solid {CARD_BORDER};
#   border-radius: 6px; padding: 4px 11px; letter-spacing: 1.5px;
#   background: {CARD_BG};
# }}
# .topbar-dot {{
#   width: 8px; height: 8px; border-radius: 50%;
#   background: #22C55E; box-shadow: 0 0 8px rgba(34,197,94,0.5);
#   animation: heartbeat 2s infinite;
# }}
# @keyframes heartbeat {{
#   0%,100% {{ transform: scale(1); opacity: 1; }}
#   50% {{ transform: scale(0.8); opacity: 0.5; }}
# }}

# /* ── section label ── */
# .sec-label {{
#   font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
#   letter-spacing: 2.5px; text-transform: uppercase; color: {TEXT_MUT};
#   margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
# }}
# .sec-label::after {{ content: ''; flex: 1; height: 1px; background: {CARD_BORDER}; }}

# /* ── card ── */
# .card {{
#   background: {CARD_BG};
#   border: 1px solid {CARD_BORDER};
#   border-radius: 12px; padding: 14px 16px;
#   box-shadow: 0 2px 12px rgba(0,0,0,0.06);
#   transition: box-shadow 0.25s;
# }}
# .card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.10); }}

# /* ── status card ── */
# .status-wrap {{
#   border-radius: 12px; padding: 18px 16px 16px;
#   text-align: center; position: relative; overflow: hidden;
#   border: 1px solid; transition: all 0.4s ease;
#   box-shadow: 0 2px 16px rgba(0,0,0,0.07);
# }}
# .status-wrap::before {{
#   content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
# }}
# .s-awake   {{ border-color: rgba(34,197,94,.3);  background: {'rgba(34,197,94,.06)' if ss.dark_mode else 'rgba(34,197,94,.05)'}; }}
# .s-awake::before  {{ background: linear-gradient(90deg,#22C55E,#16A34A); }}
# .s-awake   .s-value {{ color: #16A34A; }}
# .s-tired   {{ border-color: rgba(234,179,8,.35);  background: {'rgba(234,179,8,.06)' if ss.dark_mode else 'rgba(234,179,8,.05)'}; }}
# .s-tired::before  {{ background: linear-gradient(90deg,#EAB308,#CA8A04); }}
# .s-tired   .s-value {{ color: #CA8A04; }}
# .s-drowsy  {{ border-color: rgba(249,115,22,.35); background: {'rgba(249,115,22,.06)' if ss.dark_mode else 'rgba(249,115,22,.05)'}; }}
# .s-drowsy::before {{ background: linear-gradient(90deg,#F97316,#EA580C); }}
# .s-drowsy  .s-value {{ color: #EA580C; }}
# .s-critical {{ border-color: rgba(239,68,68,.35); background: {'rgba(239,68,68,.06)' if ss.dark_mode else 'rgba(239,68,68,.05)'}; }}
# .s-critical::before {{ background: linear-gradient(90deg,#EF4444,#DC2626); }}
# .s-critical .s-value {{ color: #DC2626; }}
# .s-phone   {{ border-color: rgba(239,68,68,.35);  background: {'rgba(239,68,68,.06)' if ss.dark_mode else 'rgba(239,68,68,.05)'}; }}
# .s-phone::before  {{ background: linear-gradient(90deg,#EF4444,#DC2626); }}
# .s-phone   .s-value {{ color: #DC2626; }}

# .s-icon  {{ font-size: 2rem; margin-bottom: 5px; }}
# .s-value {{
#   font-family: 'IM Fell English', 'Times New Roman', serif;
#   font-size: 1.6rem; font-weight: 400; font-style: italic;
#   letter-spacing: 1px; margin: 4px 0;
# }}
# .s-desc {{
#   font-family: 'Lora', 'Times New Roman', serif;
#   font-size: 0.7rem; color: {TEXT_SEC}; margin-top: 4px;
#   font-style: italic;
# }}

# /* ── metric tile ── */
# .metric-tile {{
#   background: {CARD_BG};
#   border: 1px solid {CARD_BORDER};
#   border-radius: 10px; padding: 11px 15px;
#   display: flex; align-items: center; gap: 13px;
#   margin-bottom: 8px;
#   box-shadow: 0 1px 6px rgba(0,0,0,0.05);
#   transition: transform 0.2s, box-shadow 0.2s;
# }}
# .metric-tile:hover {{ transform: translateX(3px); box-shadow: 0 3px 14px rgba(0,0,0,0.09); }}
# .mt-icon {{ font-size: 1.5rem; flex-shrink: 0; }}
# .mt-val  {{
#   font-family: 'JetBrains Mono', monospace;
#   font-size: 1.5rem; line-height: 1; font-weight: 500;
# }}
# .mt-key  {{
#   font-family: 'Lora', 'Times New Roman', serif;
#   font-size: 0.62rem; color: {TEXT_MUT};
#   letter-spacing: 1px; text-transform: uppercase; margin-top: 3px;
# }}

# /* ── alert log ── */
# .alert-row {{
#   display: flex; align-items: flex-start; gap: 9px;
#   padding: 7px 11px; border-radius: 7px; margin-bottom: 5px;
#   border-left: 3px solid; font-size: 0.74rem; line-height: 1.4;
#   font-family: 'Lora', 'Times New Roman', serif;
#   transition: opacity 0.3s;
# }}
# .ar-ok     {{ background: rgba(34,197,94,.06);  border-color: #22C55E; color: {'#86EFAC' if ss.dark_mode else '#15803D'}; }}
# .ar-warn   {{ background: rgba(234,179,8,.07);  border-color: #EAB308; color: {'#FDE047' if ss.dark_mode else '#92400E'}; }}
# .ar-danger {{ background: rgba(239,68,68,.07);  border-color: #EF4444; color: {'#FCA5A5' if ss.dark_mode else '#991B1B'}; }}
# .ar-time   {{
#   font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
#   color: {TEXT_MUT}; margin-left: auto; flex-shrink: 0; padding-top: 2px;
# }}

# /* ── chart toggle buttons ── */
# .chart-btn {{
#   font-family: 'Lora', 'Times New Roman', serif;
#   font-size: 0.72rem; padding: 5px 14px;
#   border-radius: 6px; border: 1px solid {CARD_BORDER};
#   cursor: pointer; transition: all 0.2s;
#   background: {CARD_BG}; color: {TEXT_SEC};
#   letter-spacing: 0.5px;
# }}
# .chart-btn.active {{
#   background: {ACCENT}; color: white; border-color: {ACCENT};
#   box-shadow: 0 2px 8px rgba(3,105,161,0.25);
# }}

# /* ── dark mode toggle ── */
# .dm-toggle {{
#   font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
#   padding: 5px 12px; border-radius: 20px;
#   border: 1px solid {CARD_BORDER};
#   background: {CARD_BG}; color: {TEXT_SEC};
#   cursor: pointer; letter-spacing: 1px;
# }}

# /* ── streamlit overrides ── */
# [data-testid="stImage"] {{ line-height: 0; }}
# [data-testid="stImage"] img {{ border-radius: 10px; width: 100%; display: block; }}
# .js-plotly-plot .plotly,
# .js-plotly-plot .plotly div {{ background: transparent !important; }}
# div[data-testid="stHorizontalBlock"] {{ gap: 12px !important; }}
# .stButton > button {{
#   font-family: 'Lora', 'Times New Roman', serif !important;
#   font-size: 0.75rem !important; border-radius: 8px !important;
#   padding: 5px 16px !important; transition: all 0.2s !important;
#   border: 1px solid {CARD_BORDER} !important;
#   background: {CARD_BG} !important; color: {TEXT_PRI} !important;
#   letter-spacing: 0.5px !important;
# }}
# .stButton > button:hover {{
#   background: {ACCENT} !important; color: white !important;
#   border-color: {ACCENT} !important;
#   box-shadow: 0 2px 8px rgba(3,105,161,0.3) !important;
# }}

# /* animate status change */
# @keyframes statusPop {{
#   0% {{ transform: scale(0.97); opacity: 0.7; }}
#   100% {{ transform: scale(1); opacity: 1; }}
# }}
# .status-wrap {{ animation: statusPop 0.3s ease; }}
# </style>
# """, unsafe_allow_html=True)

# # ──────────────────────────────────────────────────────────────────────────────
# # MODEL LOADING
# # ──────────────────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_models():
#     eye_m   = tf.keras.models.load_model("models/eye_state_cnn.h5")
#     mouth_m = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")
#     yolo_m  = YOLO("yolov8n.pt")
#     fm      = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
#     return eye_m, mouth_m, yolo_m, fm

# eye_model, mouth_model, yolo_model, face_mesh = load_models()

# # ──────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ──────────────────────────────────────────────────────────────────────────────
# LEFT_EYE  = [33,133,160,159,158,144]
# MOUTH_IDX = [61,291,0,17,78,308]
# IMG=64; PAD=16
# EYE_THRESH=2.0; MOUTH_FRAMES=8
# OBSERVE_TIME=20; ALERT_HOLD=8

# def preprocess(img):
#     return np.expand_dims(cv2.resize(img,(IMG,IMG))/255.0, 0)

# # ──────────────────────────────────────────────────────────────────────────────
# # BACKGROUND THREAD
# # ──────────────────────────────────────────────────────────────────────────────
# def bg_worker(q: queue.Queue, stop_event: threading.Event):
#     cap = cv2.VideoCapture(0)
#     eye_close_start = None
#     eye_event_triggered = False
#     mouth_open_frames = 0
#     long_blinks = 0
#     yawn_count  = 0
#     window_start = None
#     alert_start  = None
#     state = "AWAKE"
#     alert_log = []
#     hist = {"TIRED":0,"DROWSY":0,"CRITICAL":0,"PHONE":0,"_phone_last":False}
#     total_alerts = 0
#     session_start = time.time()

#     def push_log(msg, level="warn"):
#         nonlocal total_alerts
#         alert_log.insert(0,{"msg":msg,"level":level,"time":time.strftime("%H:%M:%S")})
#         if len(alert_log)>8: alert_log.pop()
#         total_alerts += 1

#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.05); continue

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = face_mesh.process(rgb)
#         eye_closed = yawning = False
#         now = time.time()

#         if res.multi_face_landmarks:
#             lm = res.multi_face_landmarks[0]
#             ex=[int(lm.landmark[i].x*w) for i in LEFT_EYE]
#             ey=[int(lm.landmark[i].y*h) for i in LEFT_EYE]
#             ec=frame[max(0,min(ey)-PAD):max(ey)+PAD, max(0,min(ex)-PAD):max(ex)+PAD]
#             mx=[int(lm.landmark[i].x*w) for i in MOUTH_IDX]
#             my=[int(lm.landmark[i].y*h) for i in MOUTH_IDX]
#             mc=frame[max(0,min(my)-PAD):max(my)+PAD, max(0,min(mx)-PAD):max(mx)+PAD]
#             if ec.size: eye_closed = eye_model.predict(preprocess(ec),verbose=0)[0][0]<0.5
#             if mc.size: yawning    = mouth_model.predict(preprocess(mc),verbose=0)[0][0]<0.5

#         if eye_closed:
#             if eye_close_start is None: eye_close_start=now
#             if now-eye_close_start>=EYE_THRESH and not eye_event_triggered:
#                 long_blinks+=1; eye_event_triggered=True
#                 push_log("Long eye closure detected","warn")
#         else:
#             eye_close_start=None; eye_event_triggered=False

#         if yawning:
#             mouth_open_frames+=1
#         else:
#             if mouth_open_frames>=MOUTH_FRAMES:
#                 yawn_count+=1; push_log("Yawn detected","warn")
#             mouth_open_frames=0

#         if window_start is None and (long_blinks>0 or yawn_count>0):
#             window_start=now

#         if window_start:
#             if now-window_start>OBSERVE_TIME:
#                 long_blinks=yawn_count=0; window_start=None; state="AWAKE"
#             if long_blinks>=1 or yawn_count>=2:
#                 if state!="TIRED": push_log("Driver → TIRED","warn"); hist["TIRED"]+=1
#                 state="TIRED"; alert_start=now
#             if long_blinks>=1 and yawn_count>=2:
#                 if state!="DROWSY": push_log("Driver → DROWSY","danger"); hist["DROWSY"]+=1
#                 state="DROWSY"; alert_start=now
#             if long_blinks>=3:
#                 if state!="CRITICAL": push_log("⚠ CRITICAL fatigue!","danger"); hist["CRITICAL"]+=1
#                 state="CRITICAL"; alert_start=now

#         if alert_start and now-alert_start>ALERT_HOLD:
#             alert_start=window_start=None; long_blinks=yawn_count=0; state="AWAKE"

#         phone_detected=False
#         for r in yolo_model(frame, verbose=False):
#             for box in r.boxes:
#                 if yolo_model.names[int(box.cls[0])]=="cell phone":
#                     phone_detected=True
#                     x1,y1,x2,y2=map(int,box.xyxy[0])
#                     cv2.rectangle(frame,(x1,y1),(x2,y2),(220,50,50),2)
#                     cv2.putText(frame,"PHONE",(x1,y1-8),
#                                 cv2.FONT_HERSHEY_SIMPLEX,.65,(220,50,50),2)

#         if phone_detected and not hist["_phone_last"]:
#             push_log("📵 Phone usage detected!","danger"); hist["PHONE"]+=1
#         hist["_phone_last"]=phone_detected

#         driver_alert(state, phone_detected)

#         # frame overlay
#         oc_map={"AWAKE":(34,197,94),"TIRED":(234,179,8),
#                 "DROWSY":(249,115,22),"CRITICAL":(239,68,68)}
#         oc=oc_map.get(state,(34,197,94))
#         ov=frame.copy()
#         cv2.rectangle(ov,(0,h-46),(w,h),(20,30,50),-1)
#         cv2.addWeighted(ov,.5,frame,.5,0,frame)
#         cv2.putText(frame,state,(10,h-14),cv2.FONT_HERSHEY_SIMPLEX,.75,oc,2)
#         cv2.putText(frame,f"B:{long_blinks}  Y:{yawn_count}",
#                     (w-160,h-14),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,200,215),1)
#         if phone_detected:
#             cv2.putText(frame,"PHONE DETECTED",(10,36),
#                         cv2.FONT_HERSHEY_SIMPLEX,.7,(239,68,68),2)

#         data=dict(
#             frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
#             state=state, phone=phone_detected,
#             blinks=long_blinks, yawns=yawn_count,
#             log=list(alert_log),
#             hist={k:v for k,v in hist.items() if not k.startswith("_")},
#             total_alerts=total_alerts,
#             elapsed=int(now-session_start),
#         )
#         try:
#             q.put_nowait(data)
#         except queue.Full:
#             try: q.get_nowait()
#             except queue.Empty: pass
#             q.put_nowait(data)

#     cap.release()

# # ──────────────────────────────────────────────────────────────────────────────
# # START BG THREAD ONCE
# # ──────────────────────────────────────────────────────────────────────────────
# if ss.frame_queue is None:
#     ss.frame_queue = queue.Queue(maxsize=2)
#     ss._stop_event = threading.Event()
#     ss.bg_thread   = threading.Thread(
#         target=bg_worker, args=(ss.frame_queue, ss._stop_event), daemon=True)
#     ss.bg_thread.start()

# # ──────────────────────────────────────────────────────────────────────────────
# # RENDER HELPERS
# # ──────────────────────────────────────────────────────────────────────────────
# _STATUS_META={
#     "AWAKE":   ("s-awake",   "😊","Driver is alert and focused"),
#     "TIRED":   ("s-tired",   "😑","Mild fatigue — monitor closely"),
#     "DROWSY":  ("s-drowsy",  "😴","High fatigue — recommend a break"),
#     "CRITICAL":("s-critical","⚠️","Extreme fatigue — stop the vehicle now"),
#     "PHONE":   ("s-phone",   "📵","Phone in use — eyes off the road"),
# }

# def status_html(state, phone):
#     key="PHONE" if phone else state
#     cls,icon,desc=_STATUS_META.get(key,_STATUS_META["AWAKE"])
#     return f"""<div class="status-wrap {cls}">
#       <div class="s-icon">{icon}</div>
#       <div class="s-value">{key}</div>
#       <div class="s-desc">{desc}</div>
#     </div>"""

# def metrics_html(blinks, yawns, phone):
#     pc="#DC2626" if phone else "#16A34A"
#     pv="ACTIVE" if phone else "CLEAR"
#     return f"""<div>
#       <div class="metric-tile">
#         <div class="mt-icon">👁</div>
#         <div><div class="mt-val" style="color:#0369A1">{blinks}</div>
#              <div class="mt-key">Long Blinks</div></div>
#       </div>
#       <div class="metric-tile">
#         <div class="mt-icon">🥱</div>
#         <div><div class="mt-val" style="color:#CA8A04">{yawns}</div>
#              <div class="mt-key">Yawns Detected</div></div>
#       </div>
#       <div class="metric-tile">
#         <div class="mt-icon">📱</div>
#         <div><div class="mt-val" style="color:{pc}">{pv}</div>
#              <div class="mt-key">Phone Detection</div></div>
#       </div>
#     </div>"""

# def alerts_html(log):
#     if not log:
#         return f'<div style="color:{TEXT_MUT};font-size:.78rem;padding:10px;font-family:Lora,serif;font-style:italic">No alerts recorded yet.</div>'
#     _c={"ok":"ar-ok","warn":"ar-warn","danger":"ar-danger"}
#     out=""
#     for item in log:
#         c=_c.get(item["level"],"ar-warn")
#         out+=f'<div class="alert-row {c}"><span>{item["msg"]}</span>' \
#              f'<span class="ar-time">{item["time"]}</span></div>'
#     return out

# CHART_COLORS=["#EAB308","#F97316","#EF4444","#A855F7"]

# def make_bar_chart(hist):
#     labels=["TIRED","DROWSY","CRITICAL","PHONE"]
#     values=[hist.get(k,0) for k in labels]
#     fig=go.Figure(go.Bar(
#         x=labels, y=values,
#         marker_color=CHART_COLORS,
#         marker_line_color="rgba(0,0,0,0)",
#         text=values, textposition="outside",
#         textfont=dict(family="JetBrains Mono",size=11,color=LABEL_COLOR),
#     ))
#     fig.update_layout(
#         paper_bgcolor=PLOT_PAPER,
#         plot_bgcolor=PLOT_PLOT,
#         font=dict(family="Lora, Times New Roman, serif",color=LABEL_COLOR),
#         margin=dict(l=8,r=8,t=10,b=8),
#         height=175,
#         xaxis=dict(showgrid=False,
#                    tickfont=dict(size=10,color=LABEL_COLOR,family="Lora, serif"),
#                    linecolor=GRID_COLOR),
#         yaxis=dict(showgrid=True,gridcolor=GRID_COLOR,
#                    zeroline=False,tickfont=dict(size=9,color=TICK_COLOR)),
#         bargap=0.38,
#     )
#     return fig

# def make_pie_chart(hist):
#     labels=["TIRED","DROWSY","CRITICAL","PHONE"]
#     values=[hist.get(k,0) for k in labels]
#     total=sum(values)
#     if total==0: values=[1,1,1,1]   # placeholder when no events yet
#     fig=go.Figure(go.Pie(
#         labels=labels, values=values,
#         marker_colors=CHART_COLORS,
#         textinfo="label+percent",
#         textfont=dict(family="Lora, Times New Roman, serif",size=10,color=TEXT_PRI),
#         hole=0.42,
#         hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
#     ))
#     fig.update_layout(
#         paper_bgcolor=PLOT_PAPER,
#         font=dict(family="Lora, Times New Roman, serif",color=LABEL_COLOR),
#         margin=dict(l=8,r=8,t=10,b=8),
#         height=175,
#         showlegend=False,
#         annotations=[dict(
#             text=f"<b>{total}</b>" if total>0 else "—",
#             x=0.5,y=0.5,font_size=18,showarrow=False,
#             font=dict(family="JetBrains Mono",color=TEXT_PRI)
#         )],
#     )
#     return fig

# def sysinfo_html(total, elapsed):
#     h=elapsed//3600; m=(elapsed%3600)//60; s=elapsed%60
#     return f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.64rem;
#                            color:{TEXT_MUT};line-height:2;padding:2px 0">
#       <div>UPTIME &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:{ACCENT}">{h:02d}h {m:02d}m {s:02d}s</span></div>
#       <div>TOTAL ALERTS <span style="color:#EF4444">{total}</span></div>
#       <div>EYE THRESHOLD <span style="color:{ACCENT}">2.0 s</span></div>
#       <div>YAWN FRAMES &nbsp;<span style="color:{ACCENT}">8</span></div>
#       <div>OBSERVE WIN &nbsp;<span style="color:{ACCENT}">20 s</span></div>
#       <div>ALERT HOLD &nbsp; <span style="color:{ACCENT}">8 s</span></div>
#     </div>"""

# # ──────────────────────────────────────────────────────────────────────────────
# # TOP BAR
# # ──────────────────────────────────────────────────────────────────────────────
# latest    = ss.latest
# elapsed_s = latest.get("elapsed", 0)
# h2=elapsed_s//3600; m2=(elapsed_s%3600)//60; s2=elapsed_s%60

# dm_icon = "☀️ Light" if ss.dark_mode else "🌙 Dark"

# tb_l, tb_mid, tb_r = st.columns([3, 1, 1.2])
# with tb_l:
#     st.markdown(f"""
#     <div class="topbar">
#       <div class="topbar-logo">🚗</div>
#       <div>
#         <div class="topbar-title">Driver Monitoring System</div>
#         <div class="topbar-sub">Real-Time Fatigue &amp; Distraction Detection</div>
#       </div>
#       <div class="topbar-right">
#         <div class="topbar-clock">⏱ {h2:02d}:{m2:02d}:{s2:02d}</div>
#         <div class="topbar-dot"></div>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

# with tb_r:
#     st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if st.button(dm_icon, key="dm_btn"):
#             ss.dark_mode = not ss.dark_mode
#             st.rerun()
#     with c2:
#         if st.button("📊 Bar", key="bar_btn"):
#             ss.chart_type = "bar"
#     with c3:
#         if st.button("🥧 Pie", key="pie_btn"):
#             ss.chart_type = "pie"

# st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# # ──────────────────────────────────────────────────────────────────────────────
# # MAIN LAYOUT  — fixed height rows, no scroll
# # ──────────────────────────────────────────────────────────────────────────────
# _, main, _ = st.columns([0.01, 0.98, 0.01])
# with main:
#     col_cam, col_mid, col_right = st.columns([2.0, 1.3, 1.3], gap="medium")

#     with col_cam:
#         st.markdown('<div class="sec-label">📷 Live Camera Feed</div>',
#                     unsafe_allow_html=True)
#         cam_ph = st.empty()
#         st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
#         chart_label = "📊 Event Frequency — Bar Chart" if ss.chart_type=="bar" else "🥧 Event Distribution — Pie Chart"
#         st.markdown(f'<div class="sec-label">{chart_label}</div>', unsafe_allow_html=True)
#         chart_ph = st.empty()

#     with col_mid:
#         st.markdown('<div class="sec-label">🧠 Driver Status</div>',
#                     unsafe_allow_html=True)
#         status_ph = st.empty()
#         st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">📈 Fatigue Metrics</div>',
#                     unsafe_allow_html=True)
#         metrics_ph = st.empty()

#     with col_right:
#         st.markdown('<div class="sec-label">🔔 System Alerts</div>',
#                     unsafe_allow_html=True)
#         alerts_ph = st.empty()
#         st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">⚙️ System Info</div>',
#                     unsafe_allow_html=True)
#         sysinfo_ph = st.empty()

# # ──────────────────────────────────────────────────────────────────────────────
# # LIVE RENDER LOOP
# # ──────────────────────────────────────────────────────────────────────────────
# _counter = 0

# while True:
#     try:
#         data = ss.frame_queue.get(timeout=0.5)
#     except queue.Empty:
#         continue

#     ss.latest = data
#     _counter += 1

#     # camera
#     cam_ph.image(data["frame"], channels="RGB", width="stretch")

#     # chart — bar or pie
#     fig = make_bar_chart(data["hist"]) if ss.chart_type=="bar" else make_pie_chart(data["hist"])
#     chart_ph.plotly_chart(
#         fig,
#         width="stretch",
#         config={"displayModeBar": False},
#         key=f"chart_{_counter}",
#     )

#     # status
#     status_ph.markdown(status_html(data["state"], data["phone"]),
#                        unsafe_allow_html=True)

#     # metrics
#     metrics_ph.markdown(metrics_html(data["blinks"], data["yawns"], data["phone"]),
#                         unsafe_allow_html=True)

#     # alerts
#     alerts_ph.markdown(
#         f'<div class="card" style="padding:11px 13px;max-height:230px;overflow-y:auto">'
#         + alerts_html(data["log"]) + '</div>',
#         unsafe_allow_html=True,
#     )

#     # sysinfo
#     sysinfo_ph.markdown(
#         '<div class="card">' + sysinfo_html(data["total_alerts"], data["elapsed"]) + '</div>',
#         unsafe_allow_html=True,
#     )

# dashboard.py  ──  Driver Monitoring System  ──  Module 10
# Run:  streamlit run dashboard.py

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import threading
import queue
import plotly.graph_objects as go
from ultralytics import YOLO
from voice_alert import driver_alert
import io
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        frame_queue=None, _stop_event=None,
        bg_thread=None, latest={},
        session_start=time.time(),
        dark_mode=False,
        chart_type="bar",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ss = st.session_state

# ──────────────────────────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────────────────────────
if ss.dark_mode:
    BG          = "#0F172A"
    CARD_BG     = "#1E293B"
    CARD_BDR    = "rgba(100,160,220,0.18)"
    TEXT_PRI    = "#E2EAF4"
    TEXT_SEC    = "#6A8FAA"
    TEXT_MUT    = "#304A60"
    ACCENT      = "#38BDF8"
    TOPBAR_BG   = "rgba(15,23,42,0.98)"
    TOPBAR_BDR  = "rgba(56,189,248,0.2)"
    PLOT_BG     = "rgba(0,0,0,0)"
    GRID        = "rgba(56,189,248,0.08)"
    TICK        = "#2A4560"
    LABEL       = "#5A8AA4"
    INPUT_BG    = "#1E293B"
    SHADOW      = "rgba(0,0,0,0.35)"
else:
    BG          = "#EEF2F7"
    CARD_BG     = "#FFFFFF"
    CARD_BDR    = "rgba(100,150,200,0.2)"
    TEXT_PRI    = "#1A2940"
    TEXT_SEC    = "#4A6A88"
    TEXT_MUT    = "#9AB0C4"
    ACCENT      = "#0369A1"
    TOPBAR_BG   = "rgba(255,255,255,0.98)"
    TOPBAR_BDR  = "rgba(100,160,220,0.25)"
    PLOT_BG     = "rgba(255,255,255,0)"
    GRID        = "rgba(100,160,220,0.12)"
    TICK        = "#A0B8CC"
    LABEL       = "#5A7A94"
    INPUT_BG    = "#FFFFFF"
    SHADOW      = "rgba(0,0,0,0.07)"

# ──────────────────────────────────────────────────────────────────────────────
# CSS  — single viewport, no scroll, clean grid
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── viewport lock ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html {{ height: 100vh; overflow: hidden; }}
body {{ height: 100vh; overflow: hidden; margin: 0; background: {BG}; }}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] > div > div {{
  background: {BG} !important;
  color: {TEXT_PRI};
  font-family: 'Lora', 'Times New Roman', Georgia, serif;
}}

/* kill ALL streamlit chrome */
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], footer, #MainMenu,
[data-testid="stStatusWidget"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* main wrapper — exact viewport */
.block-container {{
  padding: 0 !important;
  max-width: 100% !important;
  overflow: hidden !important;
}}

/* columns tight */
[data-testid="stHorizontalBlock"] {{
  gap: 10px !important;
  align-items: stretch !important;
}}
[data-testid="column"] {{
  padding: 0 !important;
  overflow: hidden;
}}

/* ── top bar ── */
.topbar {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 20px;
  background: {TOPBAR_BG};
  border-bottom: 1px solid {TOPBAR_BDR};
  backdrop-filter: blur(16px);
  box-shadow: 0 1px 10px {SHADOW};
  height: 54px;
}}
.topbar-logo {{
  width: 36px; height: 36px; border-radius: 8px; flex-shrink: 0;
  background: linear-gradient(135deg,#0EA5E9,#0369A1);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  box-shadow: 0 2px 8px rgba(14,165,233,0.35);
}}
.topbar-title {{
  font-family: 'IM Fell English', 'Times New Roman', serif;
  font-style: bold; font-size: 1.05rem; color: {TEXT_PRI}; margin: 0;
}}
.topbar-sub {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.54rem;
  color: {TEXT_SEC}; letter-spacing: 2px; text-transform: uppercase; margin-top: 1px;
}}
.topbar-spacer {{ flex: 1; }}
.topbar-clock {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
  color: {ACCENT}; border: 1px solid {CARD_BDR};
  border-radius: 6px; padding: 4px 10px; letter-spacing: 1.5px;
  background: {CARD_BG};
}}
.topbar-dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: #22C55E; box-shadow: 0 0 7px rgba(34,197,94,0.6);
  animation: beat 2s infinite;
  flex-shrink: 0;
}}
@keyframes beat {{
  0%,100%{{transform:scale(1);opacity:1}} 50%{{transform:scale(.75);opacity:.5}}
}}

/* ── main grid wrapper ── */
.main-grid {{
  display: grid;
  grid-template-columns: 2fr 1.25fr 1.25fr;
  gap: 10px;
  padding: 10px 16px;
  height: calc(100vh - 54px);
  overflow: hidden;
}}

/* ── panel (column container) ── */
.panel {{
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: hidden;
  height: 100%;
}}

/* ── section label ── */
.sec-label {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.56rem;
  letter-spacing: 2.5px; text-transform: uppercase; color: {TEXT_MUT};
  display: flex; align-items: center; gap: 6px; margin: 0;
  flex-shrink: 0;
}}
.sec-label::after {{
  content: ''; flex: 1; height: 1px; background: {CARD_BDR};
}}

/* ── card ── */
.card {{
  background: {CARD_BG};
  border: 1px solid {CARD_BDR};
  border-radius: 11px;
  padding: 12px 15px;
  box-shadow: 0 2px 10px {SHADOW};
  transition: box-shadow 0.2s;
  overflow: hidden;
}}

/* ── status card ── */
.status-wrap {{
  border-radius: 11px; padding: 16px 14px 14px;
  text-align: center; position: relative; overflow: hidden;
  border: 1px solid; flex-shrink: 0;
  box-shadow: 0 2px 12px {SHADOW};
  animation: pop 0.3s ease;
}}
@keyframes pop {{ 0%{{transform:scale(.97);opacity:.7}} 100%{{transform:scale(1);opacity:1}} }}
.status-wrap::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}}
.s-awake   {{ border-color:rgba(34,197,94,.3);  background:{'rgba(34,197,94,.07)' if ss.dark_mode else 'rgba(240,255,245,.9)'}; }}
.s-awake::before   {{ background:linear-gradient(90deg,#22C55E,#16A34A); }}
.s-awake   .s-val  {{ color:#16A34A; }}
.s-tired   {{ border-color:rgba(234,179,8,.35); background:{'rgba(234,179,8,.07)' if ss.dark_mode else 'rgba(255,253,235,.9)'}; }}
.s-tired::before   {{ background:linear-gradient(90deg,#EAB308,#CA8A04); }}
.s-tired   .s-val  {{ color:#CA8A04; }}
.s-drowsy  {{ border-color:rgba(249,115,22,.35);background:{'rgba(249,115,22,.07)' if ss.dark_mode else 'rgba(255,247,237,.9)'}; }}
.s-drowsy::before  {{ background:linear-gradient(90deg,#F97316,#EA580C); }}
.s-drowsy  .s-val  {{ color:#EA580C; }}
.s-critical{{ border-color:rgba(239,68,68,.38); background:{'rgba(239,68,68,.07)' if ss.dark_mode else 'rgba(255,242,242,.9)'}; }}
.s-critical::before{{ background:linear-gradient(90deg,#EF4444,#DC2626); }}
.s-critical .s-val {{ color:#DC2626; }}
.s-phone   {{ border-color:rgba(239,68,68,.38); background:{'rgba(239,68,68,.07)' if ss.dark_mode else 'rgba(255,242,242,.9)'}; }}
.s-phone::before   {{ background:linear-gradient(90deg,#EF4444,#DC2626); }}
.s-phone   .s-val  {{ color:#DC2626; }}

.s-icon {{ font-size: 1.8rem; margin-bottom: 4px; }}
.s-val  {{
  font-family: 'IM Fell English','Times New Roman',serif;
  font-style: italic; font-size: 1.5rem; letter-spacing: 1px; margin: 3px 0;
}}
.s-desc {{
  font-family: 'Lora','Times New Roman',serif; font-style: italic;
  font-size: 0.68rem; color: {TEXT_SEC}; margin-top: 3px;
}}

/* ── metric tile ── */
.metric-tile {{
  background: {CARD_BG};
  border: 1px solid {CARD_BDR};
  border-radius: 9px; padding: 10px 14px;
  display: flex; align-items: center; gap: 12px;
  box-shadow: 0 1px 5px {SHADOW};
  transition: transform 0.18s, box-shadow 0.18s;
  flex-shrink: 0;
}}
.metric-tile:hover {{ transform: translateX(4px); box-shadow: 0 3px 12px {SHADOW}; }}
.mt-icon {{ font-size: 1.4rem; flex-shrink: 0; }}
.mt-val  {{
  font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; line-height: 1;
}}
.mt-key  {{
  font-family: 'Lora','Times New Roman',serif;
  font-size: 0.6rem; color: {TEXT_MUT};
  letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;
}}

/* ── alert row ── */
.alert-row {{
  display: flex; align-items: flex-start; gap: 8px;
  padding: 7px 10px; border-radius: 7px; margin-bottom: 5px;
  border-left: 3px solid; font-size: 0.72rem; line-height: 1.4;
  font-family: 'Lora','Times New Roman',serif;
}}
.ar-ok     {{ background:rgba(34,197,94,.06);  border-color:#22C55E; color:{'#86EFAC' if ss.dark_mode else '#15803D'}; }}
.ar-warn   {{ background:rgba(234,179,8,.07);  border-color:#EAB308; color:{'#FDE047' if ss.dark_mode else '#92400E'}; }}
.ar-danger {{ background:rgba(239,68,68,.07);  border-color:#EF4444; color:{'#FCA5A5' if ss.dark_mode else '#991B1B'}; }}
.ar-time   {{
  font-family:'JetBrains Mono',monospace; font-size:0.56rem;
  color:{TEXT_MUT}; margin-left:auto; flex-shrink:0; padding-top:2px;
}}

/* ── sysinfo ── */
.sysinfo {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
  color: {TEXT_MUT}; line-height: 1.9;
}}
.sysinfo span {{ color: {ACCENT}; }}
.sysinfo .red {{ color: #EF4444; }}

/* ── streamlit element fixes ── */
[data-testid="stImage"] {{ line-height: 0; flex-shrink: 0; }}
[data-testid="stImage"] img {{
  border-radius: 9px; width: 100%; display: block;
  max-height: 340px; object-fit: cover;
}}
.js-plotly-plot .plotly,
.js-plotly-plot .plotly div {{ background: transparent !important; }}

/* buttons */
.stButton > button {{
  font-family: 'Lora','Times New Roman',serif !important;
  font-size: 0.72rem !important; border-radius: 7px !important;
  padding: 4px 14px !important; height: 30px !important;
  border: 1px solid {CARD_BDR} !important;
  background: {CARD_BG} !important; color: {TEXT_PRI} !important;
  transition: all 0.18s !important; white-space: nowrap !important;
}}
.stButton > button:hover {{
  background: {ACCENT} !important; color: white !important;
  border-color: {ACCENT} !important;
  box-shadow: 0 2px 8px rgba(3,105,161,0.3) !important;
}}

/* plotly container height */
[data-testid="stPlotlyChart"] {{
  height: 185px !important; overflow: hidden !important;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    eye_m   = tf.keras.models.load_model("models/eye_state_cnn.h5")
    mouth_m = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")
    yolo_m  = YOLO("yolov8n.pt")
    fm      = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    return eye_m, mouth_m, yolo_m, fm

eye_model, mouth_model, yolo_model, face_mesh = load_models()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
LEFT_EYE  = [33,133,160,159,158,144]
MOUTH_IDX = [61,291,0,17,78,308]
IMG=64; PAD=16
EYE_THRESH=2.0; MOUTH_FRAMES=8
OBSERVE_TIME=20; ALERT_HOLD=8

def preprocess(img):
    return np.expand_dims(cv2.resize(img,(IMG,IMG))/255.0, 0)

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREAD
# ──────────────────────────────────────────────────────────────────────────────
def bg_worker(q: queue.Queue, stop_event: threading.Event):
    cap = cv2.VideoCapture(0)
    eye_close_start = None
    eye_event_triggered = False
    mouth_open_frames = 0
    long_blinks = 0
    yawn_count  = 0
    window_start = None
    alert_start  = None
    state = "AWAKE"
    alert_log = []
    hist = {"TIRED":0,"DROWSY":0,"CRITICAL":0,"PHONE":0,"_phone_last":False}
    total_alerts = 0
    session_start = time.time()

    def push_log(msg, level="warn"):
        nonlocal total_alerts
        alert_log.insert(0,{"msg":msg,"level":level,"time":time.strftime("%H:%M:%S")})
        if len(alert_log)>8: alert_log.pop()
        total_alerts += 1

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        eye_closed = yawning = False
        now = time.time()

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            ex=[int(lm.landmark[i].x*w) for i in LEFT_EYE]
            ey=[int(lm.landmark[i].y*h) for i in LEFT_EYE]
            ec=frame[max(0,min(ey)-PAD):max(ey)+PAD, max(0,min(ex)-PAD):max(ex)+PAD]
            mx=[int(lm.landmark[i].x*w) for i in MOUTH_IDX]
            my=[int(lm.landmark[i].y*h) for i in MOUTH_IDX]
            mc=frame[max(0,min(my)-PAD):max(my)+PAD, max(0,min(mx)-PAD):max(mx)+PAD]
            if ec.size: eye_closed = eye_model.predict(preprocess(ec),verbose=0)[0][0]<0.5
            if mc.size: yawning    = mouth_model.predict(preprocess(mc),verbose=0)[0][0]<0.5

        if eye_closed:
            if eye_close_start is None: eye_close_start=now
            if now-eye_close_start>=EYE_THRESH and not eye_event_triggered:
                long_blinks+=1; eye_event_triggered=True
                push_log("Long eye closure detected","warn")
        else:
            eye_close_start=None; eye_event_triggered=False

        if yawning:
            mouth_open_frames+=1
        else:
            if mouth_open_frames>=MOUTH_FRAMES:
                yawn_count+=1; push_log("Yawn detected","warn")
            mouth_open_frames=0

        if window_start is None and (long_blinks>0 or yawn_count>0):
            window_start=now

        if window_start:
            if now-window_start>OBSERVE_TIME:
                long_blinks=yawn_count=0; window_start=None; state="AWAKE"
            if long_blinks>=1 or yawn_count>=2:
                if state!="TIRED": push_log("Driver → TIRED","warn"); hist["TIRED"]+=1
                state="TIRED"; alert_start=now
            if long_blinks>=1 and yawn_count>=2:
                if state!="DROWSY": push_log("Driver → DROWSY","danger"); hist["DROWSY"]+=1
                state="DROWSY"; alert_start=now
            if long_blinks>=3:
                if state!="CRITICAL": push_log("⚠ CRITICAL fatigue!","danger"); hist["CRITICAL"]+=1
                state="CRITICAL"; alert_start=now

        if alert_start and now-alert_start>ALERT_HOLD:
            alert_start=window_start=None; long_blinks=yawn_count=0; state="AWAKE"

        phone_detected=False
        for r in yolo_model(frame, verbose=False):
            for box in r.boxes:
                if yolo_model.names[int(box.cls[0])]=="cell phone":
                    phone_detected=True
                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(220,60,60),2)
                    cv2.putText(frame,"PHONE",(x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,.65,(220,60,60),2)

        if phone_detected and not hist["_phone_last"]:
            push_log("📵 Phone usage detected!","danger"); hist["PHONE"]+=1
        hist["_phone_last"]=phone_detected

        driver_alert(state, phone_detected)

        oc_map={"AWAKE":(34,197,94),"TIRED":(234,179,8),
                "DROWSY":(249,115,22),"CRITICAL":(239,68,68)}
        oc=oc_map.get(state,(34,197,94))
        ov=frame.copy()
        cv2.rectangle(ov,(0,h-44),(w,h),(15,25,40),-1)
        cv2.addWeighted(ov,.5,frame,.5,0,frame)
        cv2.putText(frame,state,(10,h-13),cv2.FONT_HERSHEY_SIMPLEX,.72,oc,2)
        cv2.putText(frame,f"B:{long_blinks}  Y:{yawn_count}",
                    (w-155,h-13),cv2.FONT_HERSHEY_SIMPLEX,.48,(180,200,215),1)
        if phone_detected:
            cv2.putText(frame,"PHONE DETECTED",(10,34),
                        cv2.FONT_HERSHEY_SIMPLEX,.68,(220,60,60),2)

        data=dict(
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
            state=state, phone=phone_detected,
            blinks=long_blinks, yawns=yawn_count,
            log=list(alert_log),
            hist={k:v for k,v in hist.items() if not k.startswith("_")},
            total_alerts=total_alerts,
            elapsed=int(now-session_start),
        )
        try:
            q.put_nowait(data)
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            q.put_nowait(data)

    cap.release()

# ──────────────────────────────────────────────────────────────────────────────
# START BG THREAD ONCE
# ──────────────────────────────────────────────────────────────────────────────
if ss.frame_queue is None:
    ss.frame_queue = queue.Queue(maxsize=2)
    ss._stop_event = threading.Event()
    ss.bg_thread   = threading.Thread(
        target=bg_worker, args=(ss.frame_queue, ss._stop_event), daemon=True)
    ss.bg_thread.start()

# ──────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
_STATUS_META={
    "AWAKE":   ("s-awake",   "😊","Driver is alert and focused"),
    "TIRED":   ("s-tired",   "😑","Mild fatigue — monitor closely"),
    "DROWSY":  ("s-drowsy",  "😴","High fatigue — recommend a break"),
    "CRITICAL":("s-critical","⚠️","Extreme fatigue — stop the vehicle now"),
    "PHONE":   ("s-phone",   "📵","Phone in use — eyes off the road"),
}

def status_html(state, phone):
    key = "PHONE" if phone else state
    cls,icon,desc = _STATUS_META.get(key, _STATUS_META["AWAKE"])
    return f"""<div class="status-wrap {cls}">
      <div class="s-icon">{icon}</div>
      <div class="s-val">{key}</div>
      <div class="s-desc">{desc}</div>
    </div>"""

def metrics_html(blinks, yawns, phone):
    pc = "#DC2626" if phone else "#16A34A"
    pv = "ACTIVE" if phone else "CLEAR"
    return f"""<div>
      <div class="metric-tile">
        <span class="mt-icon">👁</span>
        <div><div class="mt-val" style="color:#0369A1">{blinks}</div>
             <div class="mt-key">Long Blinks</div></div>
      </div>
      <div class="metric-tile">
        <span class="mt-icon">🥱</span>
        <div><div class="mt-val" style="color:#CA8A04">{yawns}</div>
             <div class="mt-key">Yawns Detected</div></div>
      </div>
      <div class="metric-tile">
        <span class="mt-icon">📱</span>
        <div><div class="mt-val" style="color:{pc}">{pv}</div>
             <div class="mt-key">Phone Detection</div></div>
      </div>
    </div>"""

def alerts_html(log):
    if not log:
        return f'<p style="color:{TEXT_MUT};font-size:.72rem;font-style:italic;padding:6px 2px;margin:0">No alerts yet.</p>'
    _c = {"ok":"ar-ok","warn":"ar-warn","danger":"ar-danger"}
    out = ""
    for item in log:
        c = _c.get(item["level"],"ar-warn")
        out += f'<div class="alert-row {c}"><span>{item["msg"]}</span>' \
               f'<span class="ar-time">{item["time"]}</span></div>'
    return out

COLORS = ["#EAB308","#F97316","#EF4444","#A855F7"]

def make_bar_chart(hist):
    labels = ["TIRED","DROWSY","CRITICAL","PHONE"]
    values = [hist.get(k,0) for k in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=COLORS,
        marker_line_color="rgba(0,0,0,0)",
        text=values, textposition="outside",
        textfont=dict(family="JetBrains Mono",size=11,color=LABEL),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Lora,Times New Roman,serif",color=LABEL),
        margin=dict(l=6,r=6,t=12,b=6), height=175,
        xaxis=dict(showgrid=False,
                   tickfont=dict(size=10,color=LABEL,family="Lora,serif"),
                   linecolor=GRID),
        yaxis=dict(showgrid=True,gridcolor=GRID,zeroline=False,
                   tickfont=dict(size=9,color=TICK)),
        bargap=0.38,
    )
    return fig

def make_pie_chart(hist):
    labels = ["TIRED","DROWSY","CRITICAL","PHONE"]
    values = [hist.get(k,0) for k in labels]
    total  = sum(values)
    disp   = values if total>0 else [1,1,1,1]
    fig = go.Figure(go.Pie(
        labels=labels, values=disp,
        marker_colors=COLORS,
        textinfo="label+percent",
        textfont=dict(family="Lora,Times New Roman,serif",size=9,color=TEXT_PRI),
        hole=0.44,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(family="Lora,Times New Roman,serif",color=LABEL),
        margin=dict(l=6,r=6,t=8,b=6), height=175,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total}</b>" if total>0 else "—",
            x=0.5,y=0.5,showarrow=False,
            font=dict(family="JetBrains Mono",size=16,color=TEXT_PRI)
        )],
    )
    return fig

def sysinfo_html(total, elapsed):
    h=elapsed//3600; m=(elapsed%3600)//60; s=elapsed%60
    return f"""<div class="sysinfo">
      <div>UPTIME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>{h:02d}h {m:02d}m {s:02d}s</span></div>
      <div>TOTAL ALERTS &nbsp;<span class="red">{total}</span></div>
      <div>EYE THRESHOLD <span>2.0 s</span></div>
      <div>YAWN FRAMES &nbsp;&nbsp;<span>8</span></div>
      <div>OBSERVE WIN &nbsp;&nbsp;<span>20 s</span></div>
      <div>ALERT HOLD &nbsp;&nbsp;&nbsp;<span>8 s</span></div>
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# TOP BAR  (rendered as Streamlit columns so buttons work)
# ──────────────────────────────────────────────────────────────────────────────
latest    = ss.latest
elapsed_s = latest.get("elapsed", 0)
h2=elapsed_s//3600; m2=(elapsed_s%3600)//60; s2=elapsed_s%60
dm_label  = "☀️ Light" if ss.dark_mode else "🌙 Dark"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">🚗</div>
  <div>
    <div class="topbar-title">Driver Monitoring System</div>
    <div class="topbar-sub">Real-Time Fatigue &amp; Distraction Detection</div>
  </div>
  <div class="topbar-spacer"></div>
  <div class="topbar-clock">⏱ {h2:02d}:{m2:02d}:{s2:02d}</div>
  <div class="topbar-dot"></div>
</div>
""", unsafe_allow_html=True)

# control buttons row — tight, right-aligned
bc1, bc2, bc3, bc4 = st.columns([5.5, 0.7, 0.7, 0.7])
with bc2:
    if st.button(dm_label, key="dm_btn"):
        ss.dark_mode = not ss.dark_mode
        st.rerun()
with bc3:
    if st.button("📊 Bar", key="bar_btn"):
        ss.chart_type = "bar"
with bc4:
    if st.button("🥧 Pie", key="pie_btn"):
        ss.chart_type = "pie"

# ──────────────────────────────────────────────────────────────────────────────
# MAIN 3-COLUMN LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
col_l, col_m, col_r = st.columns([2.0, 1.25, 1.25], gap="small")

with col_l:
    st.markdown('<div class="sec-label">📷 Live Camera Feed</div>',
                unsafe_allow_html=True)
    cam_ph = st.empty()
    st.markdown('<div class="sec-label" style="margin-top:8px">📊 Event Chart</div>',
                unsafe_allow_html=True)
    chart_ph = st.empty()

with col_m:
    st.markdown('<div class="sec-label">🧠 Driver Status</div>',
                unsafe_allow_html=True)
    status_ph = st.empty()
    st.markdown('<div class="sec-label" style="margin-top:8px">📈 Fatigue Metrics</div>',
                unsafe_allow_html=True)
    metrics_ph = st.empty()

with col_r:
    st.markdown('<div class="sec-label">🔔 System Alerts</div>',
                unsafe_allow_html=True)
    alerts_ph = st.empty()
    st.markdown('<div class="sec-label" style="margin-top:8px">⚙️ System Info</div>',
                unsafe_allow_html=True)
    sysinfo_ph = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# LIVE RENDER LOOP
# ──────────────────────────────────────────────────────────────────────────────
_counter = 0

while True:
    try:
        data = ss.frame_queue.get(timeout=0.5)
    except queue.Empty:
        continue

    ss.latest = data
    _counter += 1

    # camera
    # cam_ph.image(data["frame"], channels="RGB", width="stretch")
    img_pil = Image.fromarray(data["frame"])
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    cam_ph.image(buf, width="stretch")

    
    # chart
    fig = make_bar_chart(data["hist"]) if ss.chart_type == "bar" else make_pie_chart(data["hist"])
    chart_ph.plotly_chart(
        fig, width="stretch",
        config={"displayModeBar": False},
        key=f"chart_{_counter}",
    )

    # status
    status_ph.markdown(status_html(data["state"], data["phone"]),
                       unsafe_allow_html=True)

    # metrics
    metrics_ph.markdown(metrics_html(data["blinks"], data["yawns"], data["phone"]),
                        unsafe_allow_html=True)

    # alerts
    alerts_ph.markdown(
        f'<div class="card" style="max-height:220px;overflow-y:auto;padding:10px 12px">'
        + alerts_html(data["log"]) + '</div>',
        unsafe_allow_html=True,
    )

    # sysinfo
    sysinfo_ph.markdown(
        '<div class="card">' + sysinfo_html(data["total_alerts"], data["elapsed"]) + '</div>',
        unsafe_allow_html=True,
    )


# # dashboard.py  ──  Driver Monitoring System  ──  Module 10
# # Run:  streamlit run dashboard.py

# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import time
# import threading
# import queue
# import plotly.graph_objects as go
# from ultralytics import YOLO
# from voice_alert import driver_alert

# # ──────────────────────────────────────────────────────────────────────────────
# # PAGE CONFIG
# # ──────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Driver Monitoring System",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ──────────────────────────────────────────────────────────────────────────────
# # GLOBAL CSS  —  lighter palette + refined typography
# # ──────────────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

# /* ── base ── */
# html,body,[data-testid="stAppViewContainer"]{
#   background:#0D1B2A !important;
#   color:#C9DCE8;
#   font-family:'DM Sans',sans-serif;
#   font-size:14px;
# }
# [data-testid="stAppViewContainer"]{
#   background:
#     radial-gradient(ellipse 100% 55% at 50% -5%, rgba(56,189,248,.09) 0%, transparent 65%),
#     #0D1B2A !important;
# }
# [data-testid="stHeader"],[data-testid="stToolbar"],
# footer,#MainMenu{display:none!important;}
# .block-container{padding:0!important;max-width:100%!important;}
# section[data-testid="stSidebar"]{display:none!important;}

# /* ── scrollbar ── */
# ::-webkit-scrollbar{width:4px;height:4px}
# ::-webkit-scrollbar-track{background:#111E2C}
# ::-webkit-scrollbar-thumb{background:#1E3A50;border-radius:2px}

# /* ── top bar ── */
# .topbar{
#   display:flex;align-items:center;gap:18px;
#   padding:15px 32px 13px;
#   border-bottom:1px solid rgba(56,189,248,.18);
#   background:rgba(13,27,42,.94);
#   backdrop-filter:blur(14px);
#   position:sticky;top:0;z-index:100;
# }
# .topbar-logo{
#   width:44px;height:44px;border-radius:11px;
#   background:linear-gradient(135deg,#38BDF8,#0369A1);
#   display:flex;align-items:center;justify-content:center;
#   font-size:22px;flex-shrink:0;
#   box-shadow:0 2px 12px rgba(56,189,248,.25);
# }
# .topbar-title{
#   font-family:'Syne',sans-serif;font-size:1.1rem;
#   font-weight:800;letter-spacing:3px;
#   color:#E8F4FF;text-transform:uppercase;line-height:1.15;
# }
# .topbar-sub{
#   font-family:'JetBrains Mono',monospace;font-size:.6rem;
#   color:#3A8AB0;letter-spacing:2.5px;margin-top:3px;
# }
# .topbar-right{margin-left:auto;display:flex;align-items:center;gap:14px;}
# .topbar-clock{
#   font-family:'JetBrains Mono',monospace;font-size:.72rem;
#   color:#38BDF8;border:1px solid rgba(56,189,248,.28);
#   border-radius:7px;padding:5px 13px;letter-spacing:2px;
#   background:rgba(56,189,248,.06);
# }
# .topbar-dot{
#   width:9px;height:9px;border-radius:50%;
#   background:#4ADE80;box-shadow:0 0 10px rgba(74,222,128,.6);
#   animation:pulse 2.2s infinite;
# }
# @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}

# /* ── section label ── */
# .sec-label{
#   font-family:'JetBrains Mono',monospace;font-size:.6rem;
#   letter-spacing:3px;text-transform:uppercase;color:#2E7A9E;
#   margin-bottom:10px;display:flex;align-items:center;gap:8px;
# }
# .sec-label::after{content:'';flex:1;height:1px;background:rgba(56,189,248,.1);}

# /* ── base card ── */
# .card{
#   background:rgba(255,255,255,.032);
#   border:1px solid rgba(56,189,248,.12);
#   border-radius:13px;padding:18px 20px;
# }

# /* ── status card ── */
# .status-wrap{
#   border-radius:14px;padding:26px 22px 22px;
#   text-align:center;position:relative;overflow:hidden;border:1px solid;
# }
# .status-wrap::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}

# .s-awake  {border-color:rgba(74,222,128,.28); background:rgba(74,222,128,.05);}
# .s-awake::before{background:linear-gradient(90deg,#4ADE80,#22C55E);}
# .s-awake  .s-value{color:#4ADE80;}

# .s-tired  {border-color:rgba(250,204,21,.32);  background:rgba(250,204,21,.05);}
# .s-tired::before{background:linear-gradient(90deg,#FACC15,#EAB308);}
# .s-tired  .s-value{color:#FACC15;}

# .s-drowsy {border-color:rgba(251,146,60,.32);  background:rgba(251,146,60,.05);}
# .s-drowsy::before{background:linear-gradient(90deg,#FB923C,#F97316);}
# .s-drowsy .s-value{color:#FB923C;}

# .s-critical{border-color:rgba(248,113,113,.35);background:rgba(248,113,113,.06);}
# .s-critical::before{background:linear-gradient(90deg,#F87171,#EF4444);}
# .s-critical .s-value{color:#F87171;}

# .s-phone  {border-color:rgba(248,113,113,.35); background:rgba(248,113,113,.06);}
# .s-phone::before{background:linear-gradient(90deg,#F87171,#EF4444);}
# .s-phone  .s-value{color:#F87171;}

# .s-icon {font-size:2.3rem;margin-bottom:8px;}
# .s-value{
#   font-family:'Syne',sans-serif;font-size:1.75rem;
#   font-weight:800;letter-spacing:3px;margin:5px 0;
# }
# .s-desc{font-size:.72rem;color:#4E7A90;letter-spacing:.8px;margin-top:6px;
#         font-family:'DM Sans',sans-serif;font-weight:400;}

# /* ── metric tile ── */
# .metric-tile{
#   background:rgba(255,255,255,.03);
#   border:1px solid rgba(56,189,248,.1);
#   border-radius:11px;padding:14px 18px;
#   display:flex;align-items:center;gap:15px;
#   margin-bottom:10px;
#   transition:border-color .25s;
# }
# .metric-tile:hover{border-color:rgba(56,189,248,.22);}
# .mt-icon{font-size:1.7rem;flex-shrink:0;}
# .mt-val{
#   font-family:'JetBrains Mono',monospace;
#   font-size:1.65rem;line-height:1;font-weight:500;
# }
# .mt-key{
#   font-family:'DM Sans',sans-serif;
#   font-size:.63rem;color:#365870;letter-spacing:1.8px;
#   text-transform:uppercase;margin-top:4px;
# }

# /* ── alert log ── */
# .alert-row{
#   display:flex;align-items:flex-start;gap:10px;
#   padding:9px 13px;border-radius:8px;margin-bottom:7px;
#   border-left:3px solid;font-size:.77rem;line-height:1.45;
#   font-family:'DM Sans',sans-serif;
# }
# .ar-ok    {background:rgba(74,222,128,.05); border-color:#22C55E;color:#86EFAC;}
# .ar-warn  {background:rgba(250,204,21,.06); border-color:#FACC15;color:#FDE047;}
# .ar-danger{background:rgba(248,113,113,.07);border-color:#F87171;color:#FCA5A5;}
# .ar-time  {
#   font-family:'JetBrains Mono',monospace;font-size:.6rem;
#   color:#2A5570;margin-left:auto;flex-shrink:0;padding-top:2px;
# }

# /* ── image / plotly fix ── */
# [data-testid="stImage"]{line-height:0;}
# [data-testid="stImage"] img{border-radius:11px;width:100%;display:block;}
# .js-plotly-plot .plotly,
# .js-plotly-plot .plotly div{background:transparent!important;}
# </style>
# """, unsafe_allow_html=True)

# # ──────────────────────────────────────────────────────────────────────────────
# # SESSION STATE
# # ──────────────────────────────────────────────────────────────────────────────
# def _init_state():
#     defaults = dict(
#         frame_queue=None, _stop_event=None,
#         bg_thread=None, latest={},
#         session_start=time.time(),
#     )
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# _init_state()
# ss = st.session_state

# # ──────────────────────────────────────────────────────────────────────────────
# # MODEL LOADING
# # ──────────────────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_models():
#     eye_m   = tf.keras.models.load_model("models/eye_state_cnn.h5")
#     mouth_m = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")
#     yolo_m  = YOLO("yolov8n.pt")
#     fm      = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
#     return eye_m, mouth_m, yolo_m, fm

# eye_model, mouth_model, yolo_model, face_mesh = load_models()

# # ──────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ──────────────────────────────────────────────────────────────────────────────
# LEFT_EYE  = [33,133,160,159,158,144]
# MOUTH_IDX = [61,291,0,17,78,308]
# IMG=64; PAD=16
# EYE_THRESH=2.0; MOUTH_FRAMES=8
# OBSERVE_TIME=20; ALERT_HOLD=8

# def preprocess(img):
#     return np.expand_dims(cv2.resize(img,(IMG,IMG))/255.0, 0)

# # ──────────────────────────────────────────────────────────────────────────────
# # BACKGROUND THREAD
# # ──────────────────────────────────────────────────────────────────────────────
# def bg_worker(q: queue.Queue, stop_event: threading.Event):
#     cap = cv2.VideoCapture(0)

#     eye_close_start = None
#     eye_event_triggered = False
#     mouth_open_frames = 0
#     long_blinks = 0
#     yawn_count  = 0
#     window_start = None
#     alert_start  = None
#     state = "AWAKE"
#     alert_log = []
#     hist = {"TIRED":0,"DROWSY":0,"CRITICAL":0,"PHONE":0,"_phone_last":False}
#     total_alerts = 0
#     session_start = time.time()

#     def push_log(msg, level="warn"):
#         nonlocal total_alerts
#         alert_log.insert(0,{"msg":msg,"level":level,"time":time.strftime("%H:%M:%S")})
#         if len(alert_log)>10: alert_log.pop()
#         total_alerts += 1

#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.05); continue

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = face_mesh.process(rgb)
#         eye_closed = yawning = False
#         now = time.time()

#         if res.multi_face_landmarks:
#             lm = res.multi_face_landmarks[0]
#             ex=[int(lm.landmark[i].x*w) for i in LEFT_EYE]
#             ey=[int(lm.landmark[i].y*h) for i in LEFT_EYE]
#             ec=frame[max(0,min(ey)-PAD):max(ey)+PAD, max(0,min(ex)-PAD):max(ex)+PAD]
#             mx=[int(lm.landmark[i].x*w) for i in MOUTH_IDX]
#             my=[int(lm.landmark[i].y*h) for i in MOUTH_IDX]
#             mc=frame[max(0,min(my)-PAD):max(my)+PAD, max(0,min(mx)-PAD):max(mx)+PAD]
#             if ec.size: eye_closed = eye_model.predict(preprocess(ec),verbose=0)[0][0]<0.5
#             if mc.size: yawning    = mouth_model.predict(preprocess(mc),verbose=0)[0][0]<0.5

#         if eye_closed:
#             if eye_close_start is None: eye_close_start=now
#             if now-eye_close_start>=EYE_THRESH and not eye_event_triggered:
#                 long_blinks+=1; eye_event_triggered=True
#                 push_log("Long eye closure detected","warn")
#         else:
#             eye_close_start=None; eye_event_triggered=False

#         if yawning:
#             mouth_open_frames+=1
#         else:
#             if mouth_open_frames>=MOUTH_FRAMES:
#                 yawn_count+=1; push_log("Yawn detected","warn")
#             mouth_open_frames=0

#         if window_start is None and (long_blinks>0 or yawn_count>0):
#             window_start=now

#         if window_start:
#             if now-window_start>OBSERVE_TIME:
#                 long_blinks=yawn_count=0; window_start=None; state="AWAKE"
#             if long_blinks>=1 or yawn_count>=2:
#                 if state!="TIRED": push_log("Driver → TIRED","warn"); hist["TIRED"]+=1
#                 state="TIRED"; alert_start=now
#             if long_blinks>=1 and yawn_count>=2:
#                 if state!="DROWSY": push_log("Driver → DROWSY","danger"); hist["DROWSY"]+=1
#                 state="DROWSY"; alert_start=now
#             if long_blinks>=3:
#                 if state!="CRITICAL": push_log("⚠ CRITICAL fatigue!","danger"); hist["CRITICAL"]+=1
#                 state="CRITICAL"; alert_start=now

#         if alert_start and now-alert_start>ALERT_HOLD:
#             alert_start=window_start=None; long_blinks=yawn_count=0; state="AWAKE"

#         phone_detected=False
#         for r in yolo_model(frame, verbose=False):
#             for box in r.boxes:
#                 if yolo_model.names[int(box.cls[0])]=="cell phone":
#                     phone_detected=True
#                     x1,y1,x2,y2=map(int,box.xyxy[0])
#                     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
#                     cv2.putText(frame,"PHONE",(x1,y1-8),
#                                 cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)

#         if phone_detected and not hist["_phone_last"]:
#             push_log("📵 Phone usage detected!","danger"); hist["PHONE"]+=1
#         hist["_phone_last"]=phone_detected

#         driver_alert(state, phone_detected)

#         # frame overlay
#         oc={"AWAKE":(74,222,128),"TIRED":(250,204,21),
#             "DROWSY":(251,146,60),"CRITICAL":(248,113,113)}.get(state,(74,222,128))
#         ov=frame.copy()
#         cv2.rectangle(ov,(0,h-50),(w,h),(13,27,42),-1)
#         cv2.addWeighted(ov,.55,frame,.45,0,frame)
#         cv2.putText(frame,state,(12,h-16),cv2.FONT_HERSHEY_SIMPLEX,.8,oc,2)
#         cv2.putText(frame,f"B:{long_blinks}  Y:{yawn_count}",
#                     (w-165,h-16),cv2.FONT_HERSHEY_SIMPLEX,.55,(180,200,215),1)
#         if phone_detected:
#             cv2.putText(frame,"PHONE IN USE",(12,38),
#                         cv2.FONT_HERSHEY_SIMPLEX,.75,(248,113,113),2)

#         data=dict(
#             frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
#             state=state, phone=phone_detected,
#             blinks=long_blinks, yawns=yawn_count,
#             log=list(alert_log),
#             hist={k:v for k,v in hist.items() if not k.startswith("_")},
#             total_alerts=total_alerts,
#             elapsed=int(now-session_start),
#         )
#         try:
#             q.put_nowait(data)
#         except queue.Full:
#             try: q.get_nowait()
#             except queue.Empty: pass
#             q.put_nowait(data)

#     cap.release()

# # ──────────────────────────────────────────────────────────────────────────────
# # START BG THREAD ONCE
# # ──────────────────────────────────────────────────────────────────────────────
# if ss.frame_queue is None:
#     ss.frame_queue = queue.Queue(maxsize=2)
#     ss._stop_event = threading.Event()
#     ss.bg_thread   = threading.Thread(
#         target=bg_worker, args=(ss.frame_queue, ss._stop_event), daemon=True)
#     ss.bg_thread.start()

# # ──────────────────────────────────────────────────────────────────────────────
# # RENDER HELPERS
# # ──────────────────────────────────────────────────────────────────────────────
# _STATUS_META={
#     "AWAKE":   ("s-awake",   "😊","Driver is alert and focused"),
#     "TIRED":   ("s-tired",   "😑","Mild fatigue — monitor closely"),
#     "DROWSY":  ("s-drowsy",  "😴","High fatigue — recommend a break"),
#     "CRITICAL":("s-critical","⚠️","Extreme fatigue — stop the vehicle!"),
#     "PHONE":   ("s-phone",   "📵","Phone usage — eyes off road!"),
# }

# def status_html(state, phone):
#     key="PHONE" if phone else state
#     cls,icon,desc=_STATUS_META.get(key,_STATUS_META["AWAKE"])
#     return f"""<div class="status-wrap {cls}">
#       <div class="s-icon">{icon}</div>
#       <div class="s-value">{key}</div>
#       <div class="s-desc">{desc}</div>
#     </div>"""

# def metrics_html(blinks, yawns, phone):
#     pc="#F87171" if phone else "#4ADE80"
#     pv="ACTIVE" if phone else "CLEAR"
#     return f"""
#     <div>
#       <div class="metric-tile">
#         <div class="mt-icon">👁️</div>
#         <div><div class="mt-val" style="color:#38BDF8">{blinks}</div>
#              <div class="mt-key">Long Blinks</div></div>
#       </div>
#       <div class="metric-tile">
#         <div class="mt-icon">🥱</div>
#         <div><div class="mt-val" style="color:#FACC15">{yawns}</div>
#              <div class="mt-key">Yawns Detected</div></div>
#       </div>
#       <div class="metric-tile">
#         <div class="mt-icon">📱</div>
#         <div><div class="mt-val" style="color:{pc}">{pv}</div>
#              <div class="mt-key">Phone Detection</div></div>
#       </div>
#     </div>"""

# def alerts_html(log):
#     if not log:
#         return '<div style="color:#1E4A6A;font-size:.8rem;padding:10px;font-family:\'DM Sans\',sans-serif">No alerts yet.</div>'
#     _c={"ok":"ar-ok","warn":"ar-warn","danger":"ar-danger"}
#     out=""
#     for item in log:
#         c=_c.get(item["level"],"ar-warn")
#         out+=f'<div class="alert-row {c}"><span>{item["msg"]}</span>' \
#              f'<span class="ar-time">{item["time"]}</span></div>'
#     return out

# def make_bar_chart(hist):
#     labels=["TIRED","DROWSY","CRITICAL","PHONE"]
#     values=[hist.get(k,0) for k in labels]
#     colors=["#FACC15","#FB923C","#F87171","#C084FC"]
#     fig=go.Figure(go.Bar(
#         x=labels, y=values,
#         marker_color=colors,
#         marker_line_color="rgba(0,0,0,0)",
#         text=values, textposition="outside",
#         textfont=dict(family="JetBrains Mono",size=12,color="#8BAFC4"),
#     ))
#     fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="DM Sans",color="#4A7090"),
#         margin=dict(l=10,r=10,t=14,b=10),
#         height=195,
#         xaxis=dict(
#             showgrid=False,
#             tickfont=dict(size=11,color="#5A8AA4",family="DM Sans"),
#             linecolor="rgba(56,189,248,.1)",
#         ),
#         yaxis=dict(
#             showgrid=True,gridcolor="rgba(56,189,248,.07)",
#             zeroline=False,
#             tickfont=dict(size=9,color="#2A5570"),
#         ),
#         bargap=0.38,
#     )
#     return fig

# def sysinfo_html(total, elapsed):
#     h=elapsed//3600; m=(elapsed%3600)//60; s=elapsed%60
#     return f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.67rem;
#                            color:#2A5570;line-height:2.2;padding:2px 0">
#       <div>UPTIME &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#5AA8C8">{h:02d}h {m:02d}m {s:02d}s</span></div>
#       <div>TOTAL ALERTS <span style="color:#F87171">{total}</span></div>
#       <div>EYE THRESHOLD <span style="color:#5AA8C8">2.0 s</span></div>
#       <div>YAWN FRAMES &nbsp;<span style="color:#5AA8C8">8</span></div>
#       <div>OBSERVE WIN &nbsp;<span style="color:#5AA8C8">20 s</span></div>
#       <div>ALERT HOLD &nbsp; <span style="color:#5AA8C8">8 s</span></div>
#     </div>"""

# # ──────────────────────────────────────────────────────────────────────────────
# # TOP BAR
# # ──────────────────────────────────────────────────────────────────────────────
# latest    = ss.latest
# elapsed_s = latest.get("elapsed", 0)
# h2=elapsed_s//3600; m2=(elapsed_s%3600)//60; s2=elapsed_s%60

# st.markdown(f"""
# <div class="topbar">
#   <div class="topbar-logo">🚗</div>
#   <div>
#     <div class="topbar-title">Driver Monitoring System</div>
#     <div class="topbar-sub">REAL-TIME FATIGUE &amp; DISTRACTION DETECTION</div>
#   </div>
#   <div class="topbar-right">
#     <div class="topbar-clock">⏱ {h2:02d}:{m2:02d}:{s2:02d}</div>
#     <div class="topbar-dot"></div>
#   </div>
# </div>
# <div style="height:16px"></div>
# """, unsafe_allow_html=True)

# # ──────────────────────────────────────────────────────────────────────────────
# # LAYOUT + PLACEHOLDERS
# # ──────────────────────────────────────────────────────────────────────────────
# _, main, _ = st.columns([0.015, 0.97, 0.015])
# with main:
#     col_cam, col_mid, col_right = st.columns([2.1, 1.35, 1.35], gap="medium")

#     with col_cam:
#         st.markdown('<div class="sec-label">📷 Live Camera Feed</div>',
#                     unsafe_allow_html=True)
#         cam_ph = st.empty()
#         st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">📊 Event Frequency — Session</div>',
#                     unsafe_allow_html=True)
#         chart_ph = st.empty()

#     with col_mid:
#         st.markdown('<div class="sec-label">🧠 Driver Status</div>',
#                     unsafe_allow_html=True)
#         status_ph = st.empty()
#         st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">📈 Fatigue Metrics</div>',
#                     unsafe_allow_html=True)
#         metrics_ph = st.empty()

#     with col_right:
#         st.markdown('<div class="sec-label">🔔 System Alerts</div>',
#                     unsafe_allow_html=True)
#         alerts_ph = st.empty()
#         st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
#         st.markdown('<div class="sec-label">⚙️ System Info</div>',
#                     unsafe_allow_html=True)
#         sysinfo_ph = st.empty()

# # ──────────────────────────────────────────────────────────────────────────────
# # LIVE RENDER LOOP
# # ──────────────────────────────────────────────────────────────────────────────
# _chart_counter = 0

# while True:
#     try:
#         data = ss.frame_queue.get(timeout=0.5)
#     except queue.Empty:
#         continue

#     ss.latest = data
#     _chart_counter += 1

#     # ── camera  (width="stretch" replaces use_container_width=True) ──
#     cam_ph.image(data["frame"], channels="RGB", width="stretch")

#     # ── bar chart ──
#     chart_ph.plotly_chart(
#         make_bar_chart(data["hist"]),
#         width="stretch",
#         config={"displayModeBar": False},
#         key=f"chart_{_chart_counter}",
#     )

#     # ── status ──
#     status_ph.markdown(status_html(data["state"], data["phone"]),
#                        unsafe_allow_html=True)

#     # ── metrics ──
#     metrics_ph.markdown(metrics_html(data["blinks"], data["yawns"], data["phone"]),
#                         unsafe_allow_html=True)

#     # ── alerts ──
#     alerts_ph.markdown( 
#         '<div class="card" style="padding:12px 14px;max-height:265px;overflow-y:auto">'
#         + alerts_html(data["log"]) + '</div>',
#         unsafe_allow_html=True,
#     )

#     # ── sysinfo ──
#     sysinfo_ph.markdown(
#         '<div class="card">' + sysinfo_html(data["total_alerts"], data["elapsed"]) + '</div>',
#         unsafe_allow_html=True,
#     )
    