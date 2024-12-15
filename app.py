import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the pre-trained model
model = load_model('emotion_recognition_model_tf.keras')  # Update with the correct path to your model

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to process the frame
def process_frame(frame, faceCascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        roi_gray = cv2.resize(roi_gray, (48, 48))
        face_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        face_rgb = np.expand_dims(face_rgb, axis=0)  # Add batch dimension
        face_rgb = face_rgb / 255.0  # Normalize

        prediction = model.predict(face_rgb)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Streamlit UI
st.title("Real-Time Face and Emotion Recognition")

# Initialize session state for webcam control
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Buttons for webcam control
start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

# Initialize OpenCV Cascade Classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a placeholder for the video feed
video_placeholder = st.empty()

if start_button:
    st.session_state.webcam_active = True

if stop_button:
    st.session_state.webcam_active = False

# Handle webcam feed
if st.session_state.webcam_active:
    st.text("Webcam is running. Click 'Stop Webcam' to end.")
    cap = cv2.VideoCapture(0)

    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam. Please check your camera.")
            break
        
        # Process the frame for emotion recognition
        frame = process_frame(frame, faceCascade)
        
        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update Streamlit placeholder with the current frame
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Introduce a delay to prevent high CPU usage
        time.sleep(0.03)  # Adjust based on your preference for performance

    cap.release()
    st.text("Webcam feed stopped. You can restart it by clicking 'Start Webcam'.")

else:
    st.text("Webcam is not active. Click 'Start Webcam' to begin.")

