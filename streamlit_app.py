import streamlit as st
import pandas as pd
import numpy as np
import cv2
# import streamlit_web

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# st.title('Healthtech')
#
# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)
#
# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     with mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as pose:
#         results = pose.process(frame)
#         # Draw the pose annotation on the image.
#         annotated_image = frame.copy()
#         mp_drawing.draw_landmarks(
#             annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#         annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(annotated_image)
#     # FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')

import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# faceCascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            results = pose.process(img)
            # Draw the pose annotation on the image.
            annotated_image = img.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
