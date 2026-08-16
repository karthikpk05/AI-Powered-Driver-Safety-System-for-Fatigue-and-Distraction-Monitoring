

# #voice_alert.py
# import pyttsx3
# import time
# import threading

# last_spoken_time = 0
# COOLDOWN = 6   # seconds between alerts


# def speak_async(text):

#     def run():
#         engine = pyttsx3.init()
#         engine.setProperty("rate",160)
#         engine.say(text)
#         engine.runAndWait()
#         engine.stop()

#     t = threading.Thread(target=run)
#     t.daemon = True
#     t.start()


# def driver_alert(state, phone_detected):

#     global last_spoken_time

#     now = time.time()

#     # prevent speaking too frequently
#     if now - last_spoken_time < COOLDOWN:
#         return

#     if phone_detected:
#         speak_async("Don't use mobile phone while driving")
#         last_spoken_time = now
#         return

#     if state == "TIRED":
#         speak_async("You are tired. Please take rest")
#         last_spoken_time = now

#     elif state == "DROWSY":
#         speak_async("You are very tired. Take a break")
#         last_spoken_time = now

#     elif state == "CRITICAL":
#         speak_async("Warning. You are extremely tired. Stop the vehicle immediately")
#         last_spoken_time = now

# voice_alert.py
import time
import threading

last_spoken_time = 0
COOLDOWN = 6  # seconds between alerts
_lock = threading.Lock()


def speak_async(text):
    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass  # ignore "run loop already started"
        except Exception:
            pass
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    t = threading.Thread(target=run, daemon=True)
    t.start()


def driver_alert(state, phone_detected):
    global last_spoken_time

    with _lock:
        now = time.time()
        if now - last_spoken_time < COOLDOWN:
            return

        if phone_detected:
            speak_async("Don't use mobile phone while driving")
            last_spoken_time = now
            return

        if state == "TIRED":
            speak_async("You are tired. Please take rest")
            last_spoken_time = now
        elif state == "DROWSY":
            speak_async("You are very tired. Take a break")
            last_spoken_time = now
        elif state == "CRITICAL":
            speak_async("Warning. You are extremely tired. Stop the vehicle immediately")
            last_spoken_time = now