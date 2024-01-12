#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model
import streamlit as st

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_emotion_model = load_model("jeeva_model.h5")  # Load your face emotion detection model
speech_model = load_model("speech.h5")  # Load your speech emotion detection model

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Set up the sounddevice for audio recording
duration = 3  # Adjust duration as needed
sample_rate = 22050
channels = 1

# Streamlit app
st.title("Real-time Emotion Detection")

# Open the webcam
cap = cv2.VideoCapture(0)

while st.checkbox("Run Real-time Emotion Detection"):
    ret, frame = cap.read()

    # Face detection using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image for face emotion detection
        face_roi = cv2.resize(face_roi, (224, 224))  # Resize to match the model's input shape
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Assuming the model expects RGB input
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Face emotion detection using the face emotion model
        face_emotion_prediction = face_emotion_model.predict(face_roi)
        face_emotion_label = emotion_labels[np.argmax(face_emotion_prediction)]

        # Draw bounding box and display face emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Face Emotion: {face_emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Speech emotion detection
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')  # Change dtype to float32
    sd.wait()

    mfcc = np.mean(librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)

    # Speech emotion detection using the pre-trained model
    speech_emotion_prediction = speech_model.predict(mfcc)
    speech_emotion_label = emotion_labels[np.argmax(speech_emotion_prediction)]

    # Display speech emotion
    cv2.putText(frame, f"Speech Emotion: {speech_emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

# Release the webcam
cap.release()

