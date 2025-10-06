# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# =============================
# Load Model
# =============================
@st.cache_resource
def load_fernet_model(path="models/final_model.h5"):
    model = load_model(path)
    return model

model = load_fernet_model()
st.title("🎭 Real-time Emotion Detection (FERNet-v4)")

# Emotion labels and emojis
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = ['😡','🤢','😨','😄','😢','😲','😐']

# =============================
# Helper Functions
# =============================
def preprocess_face(face_img):
    # Convert to grayscale if needed
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def draw_label(frame, text, emoji, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"{text} {emoji}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# =============================
# Sidebar Options
# =============================
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam Live"])

# =============================
# Upload Image Mode
# =============================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Face Detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected!")
        else:
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                processed_face = preprocess_face(face_img)
                prediction = model.predict(processed_face)
                label_idx = np.argmax(prediction)
                label = emotion_labels[label_idx]
                emoji = emotion_emojis[label_idx]

                draw_label(image_np, label, emoji, x, y, w, h)

            st.image(image_np, caption="Detected Emotions", use_column_width=True)

# =============================
# Webcam Live Mode with Checkbox & FPS
# =============================
elif mode == "Webcam Live":
    run_webcam = st.checkbox("Turn On Webcam")
    stframe = st.empty()
    fps_text = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        prev_time = time.time()

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                processed_face = preprocess_face(face_img)
                prediction = model.predict(processed_face)
                label_idx = np.argmax(prediction)
                label = emotion_labels[label_idx]
                emoji = emotion_emojis[label_idx]

                draw_label(frame, label, emoji, x, y, w, h)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_text.text(f"FPS: {fps:.1f}")

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        cv2.destroyAllWindows()
    else:
        st.info("Webcam is turned off. Check the box to start it.")
