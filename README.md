# ⏰ AI Alarm Clock

An intelligent alarm clock built with **Streamlit**, **OpenCV**, and **face_recognition** that only stops ringing after verifying the user’s face.  
This project enhances wake-up reliability by requiring **biometric authentication** to dismiss the alarm.

---

## 🚀 Features

### 🔒 Face Registration
- Captures 5 webcam samples  
- Generates a stable averaged 128-D face encoding  

### ⏰ Alarm Scheduling
- Set alarms in 12-hour AM/PM format  
- Alarm state saved locally and persists across pages  

### 🧠 Face Verification
- Alarm stops only after successful face match  
- Uses multi-frame verification for higher accuracy  
- Real-time webcam-based recognition  

### 🔊 Alarm Sound
- Continuous playback using `pygame.mixer`  
- Automatically stops after verification

---

## 📂 Dataset

This project does **not use any external dataset**—all data is generated from the user's webcam.

### Stored Data
- **Face Encoding:** averaged 128-D vector  
  - `data/user_face_encoding.pkl`
- **Alarm State:** saved alarm configuration  
  - `data/alarm_state.json`

**Privacy:** All facial data stays completely local to the user's device.

---

## 🤖 Model Used

- Uses the **face recognition model** to generate a 128-D facial encoding.  
- Employs **HOG/CNN-based detection** from `face_recognition` to locate faces in webcam frames.  
- During verification, the **Euclidean distance** between stored and live encodings is calculated; a match is confirmed if distance < 0.5.
    

---

## 🛠️ Technologies Used

| Component | Library |
|----------|----------|
| Web UI | Streamlit |
| Face Detection & Encoding | face_recognition (dlib) |
| Image Processing | OpenCV |
| Alarm Audio | pygame.mixer |
| Data Storage | JSON, Pickle |
| Others | NumPy, datetime |
---
