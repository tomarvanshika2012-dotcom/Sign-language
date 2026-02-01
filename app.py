from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# ---------------- CONFIG ----------------
ACTIONS = ["goodbye", "hello", "no", "please", "yes", "thanks", "sorry"]
SEQUENCE_LENGTH = 10
CONF_THRESHOLD = 0.8

# ---------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = load_model("sign_language_model.h5", compile=False)

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# Camera
cap = cv2.VideoCapture(0)

sequence = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "Waiting for gesture..."

# ---------------------------------------
def extract_keypoints(results):
    pose = np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# ---------------------------------------
def generate_frames():
    global current_prediction

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            preds = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            confidence = np.max(preds)
            action = ACTIONS[np.argmax(preds)]

            if confidence > CONF_THRESHOLD:
                current_prediction = action
            else:
                current_prediction = "Waiting for gesture..."

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(
            frame,
            current_prediction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ---------------------------------------
@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/prediction")
def prediction():
    return JSONResponse({"prediction": current_prediction})
