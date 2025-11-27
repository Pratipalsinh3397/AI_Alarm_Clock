import streamlit as st
import cv2
import numpy as np
import os
import datetime
import time
import face_recognition
import pickle
import json
from pygame import mixer


# ---------- Configuration ----------
DATA_DIR = "data"
USER_ENCODING_PATH = os.path.join(DATA_DIR, "user_face_encoding.pkl")
ALARM_STATE_PATH = os.path.join(DATA_DIR, "alarm_state.json")
ALARM_SOUND = os.path.join("resources", "sounds", "alarm-sound-1.wav")

os.makedirs(DATA_DIR, exist_ok=True)
dummy_sound_dir = os.path.dirname(ALARM_SOUND)
os.makedirs(dummy_sound_dir, exist_ok=True)
if not os.path.exists(ALARM_SOUND):
    st.warning(f"Alarm sound not found at {ALARM_SOUND}.")

try:
    mixer.init()
except Exception:
    pass


# ---------- Helper functions for Global alarm state ----------
def get_default_alarm_state():
    return {"alarm_active": False, "alarm_set_time": None}

def read_alarm_state():
    if not os.path.exists(ALARM_STATE_PATH):
        return get_default_alarm_state()
    try:
        with open(ALARM_STATE_PATH, "r") as f:
            state = json.load(f)
            if state.get("alarm_set_time"):
                state["alarm_set_time"] = datetime.datetime.fromisoformat(state["alarm_set_time"])
            return state
    except (json.JSONDecodeError, IOError):
        return get_default_alarm_state()

def write_alarm_state(state):
    try:
        state_to_write = state.copy()
        if state_to_write.get("alarm_set_time") and isinstance(state_to_write["alarm_set_time"], datetime.datetime):
            state_to_write["alarm_set_time"] = state_to_write["alarm_set_time"].isoformat()
        
        with open(ALARM_STATE_PATH, "w") as f:
            json.dump(state_to_write, f)
    except IOError as e:
        st.error(f"Could not write alarm state: {e}")

def clear_alarm_state():
    write_alarm_state(get_default_alarm_state())


# ---------- Helper Functions for alarm play and stop, face register, verify  ----------
def play_alarm():
    try:
        mixer.music.load(ALARM_SOUND)
        mixer.music.play(-1)
    except Exception as e:
        st.error(f"Error playing alarm: {e}.")

def stop_alarm():
    try:
        mixer.music.stop()
    except Exception:
        pass

def clear_user_encoding():
    if os.path.exists(USER_ENCODING_PATH):
        os.remove(USER_ENCODING_PATH)
    clear_alarm_state()

def capture_face_encoding(samples_target=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Cannot access webcam. Please close other apps using the camera.")
        return False

    frame_placeholder = st.empty()
    encodings = []
    start_time = time.time()

    while len(encodings) < samples_target and (time.time() - start_time) < 30:
        ret, frame = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)

        if faces:
            encoding = face_recognition.face_encodings(rgb, faces)[0]
            encodings.append(encoding)
            top, right, bottom, left = faces[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(frame, f"{len(encodings)}/{samples_target}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()
    frame_placeholder.empty()

    if not encodings:
        st.error("❌ No face captured. Please try again.")
        return False

    mean_encoding = np.mean(encodings, axis=0)
    with open(USER_ENCODING_PATH, "wb") as f:
        pickle.dump(mean_encoding, f)
    return True


def verify_face(tolerance, timeout=30):
    if not os.path.exists(USER_ENCODING_PATH):
        st.error("⚠️ No registered face found. Please register first.")
        return False

    with open(USER_ENCODING_PATH, "rb") as f:
        known_encoding = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Cannot access webcam.")
        return False
        
    frame_placeholder = st.empty()
    start_time = time.time()
    consistent_matches = 0
    needed_matches = 3

    while (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)

        if not faces:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            continue

        match = False

        for enc, (top, right, bottom, left) in zip(encs, faces):
            result = face_recognition.compare_faces([known_encoding], enc, tolerance=tolerance)[0]

            if result:
                consistent_matches += 1
                color = (0, 255, 0)    
                text = "Face Matched"
            else:
                consistent_matches = 0
                color = (0, 0, 255)    
                text = "Face Not Matched"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

            cv2.putText(frame, text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if consistent_matches >= needed_matches:
                match = True
                break
        
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if match:
            break

    cap.release()
    frame_placeholder.empty()
    return consistent_matches >= needed_matches


# ---------- Streamlit app ----------
st.set_page_config(page_title="AI Alarm Clock", layout="centered")
st.title("⏰ AI Alarm Clock")

for key, val in {
    "page": "register",
    "face_registered": os.path.exists(USER_ENCODING_PATH),
    "alarm_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------- Page: Register ----------
if st.session_state.page == "register":
    st.header("👤 Face Registration")
    
    if st.session_state.face_registered:
        st.success("✅ Face already registered.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Registration Data"):
                clear_user_encoding()
                st.session_state.face_registered = False
                st.rerun()
        with col2:
            st.button("Next (Set Alarm) →", type="primary",
                      on_click=lambda: st.session_state.update(page="alarm"))
    else:
        st.info("We'll capture 5 frames from your webcam to register your face.")
        if st.button("🎥 Start Registration", type="primary"):
            with st.spinner("Looking for face..."):
                success = capture_face_encoding(samples_target=5)
            if success:
                st.session_state.face_registered = True
                st.success("✅ Face registered successfully!")
                time.sleep(2)
                st.session_state.page = "alarm"
                st.rerun()


# ---------- Page: Alarm setup ----------
elif st.session_state.page == "alarm":
    st.header("⏰ Set Your Alarm")
    
    global_alarm_state = read_alarm_state()
    is_alarm_active_global = global_alarm_state["alarm_active"]
    global_alarm_time = global_alarm_state.get("alarm_set_time")

    if not is_alarm_active_global:
        st.subheader("🕒 Choose Alarm Time")
        now = datetime.datetime.now()
        current_hour_12 = now.hour % 12 or 12
        current_minute = now.minute
        current_am_pm = "AM" if now.hour < 12 else "PM"

        for k, v in {"alarm_hour": current_hour_12, "alarm_minute": current_minute, "alarm_ampm": current_am_pm}.items():
            if k not in st.session_state:
                st.session_state[k] = v

        col1, col2, col3 = st.columns(3)
        with col1: hour = st.number_input("Hour", 1, 12, key="alarm_hour")
        with col2: minute = st.number_input("Minute", 0, 59, key="alarm_minute")
        with col3: am_pm = st.selectbox("AM / PM", ["AM", "PM"], key="alarm_ampm")

        hour_24 = hour if am_pm == "AM" else (hour % 12 + 12)
        if am_pm == "AM" and hour == 12: hour_24 = 0
            
        alarm_time = now.replace(hour=hour_24, minute=minute, second=0, microsecond=0)
        if alarm_time <= now:
            alarm_time += datetime.timedelta(days=1)
        
        st.info(f"Alarm will ring at {alarm_time.strftime('%I:%M %p on %A')}")
        st.markdown("---")
        
        col_back, col_set = st.columns(2)
        with col_back:
            st.button("← Back (Face Registration)", on_click=lambda: st.session_state.update(page="register"))
        with col_set:
            if st.button("✅ Set Alarm", type="primary"):
                write_alarm_state({"alarm_active": True, "alarm_set_time": alarm_time})
                st.rerun()

    if is_alarm_active_global:
        st.success(f"✅ Alarm is set for {global_alarm_time.strftime('%I:%M %p')}")
        
        now = datetime.datetime.now()
        
        if now >= global_alarm_time:
            play_alarm()
            st.session_state.page = "verify"
            st.rerun()
        
        remaining = str(global_alarm_time - now).split(".")[0]
        st.metric(label="⏳ Time Remaining", value=remaining)
        
        if st.button("❌ Cancel Alarm"):
            clear_alarm_state()
            st.warning("🚫 Alarm cancelled.")
            time.sleep(2)
            st.rerun()
        st.markdown("---")
        
        st.button("← Back (Face Registration)", on_click=lambda: st.session_state.update(page="register"))

        time.sleep(1)
        st.rerun()


# ---------- Page: Verify ----------
elif st.session_state.page == "verify":
    st.header("🔔 WAKE UP! 😴")
    st.warning("Look at the camera to verify your identity and stop the alarm.")
       
    if st.button("🧠 Start Verification", type="primary"):
        with st.spinner("Verifying..."):
            result = verify_face(
                timeout=30, 
                tolerance=0.5 
            )
        if result:
            stop_alarm()
            clear_alarm_state()
            st.session_state.alarm_result = True
        else:
            st.session_state.alarm_result = False

        st.session_state.page = "result"
        st.rerun()


# ---------- Page: Result ----------
elif st.session_state.page == "result":
    st.header("✅ Alarm Result")
    if st.session_state.alarm_result:
        st.success("✅ Face verified — Alarm stopped. Good morning!")
        if st.button("🔄 Restart"):
            st.session_state.page = "alarm"
            st.session_state.alarm_result = None
            st.rerun()
    else:
        st.error("❌ Face not verified or timeout reached.")
        st.info("The alarm is still active. Go back to try again.")
        if st.button("🔄 Reverify"):
            st.session_state.page = "alarm"
            st.session_state.alarm_result = None
            st.rerun()