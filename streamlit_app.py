import math
from abc import ABC

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import mediapipe as mp
import av

from posture_classifier import detect_pose

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.title('Healthtech')

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.i = 0
        self.buffer = []

    @staticmethod
    def resize(image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
        return img

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.resize(img)
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            results = pose.process(img)
            detected_pose = detect_pose(results)
            st.write(detected_pose)
            print(detected_pose)

            self.buffer.append(detected_pose)
            if detected_pose is not None:
                if len(self.buffer) > 2:
                    if detected_pose == 'standing' and (
                            self.buffer[-2] == 'sitting' or self.buffer[-2] == 'floor' or self.buffer[-2] == 'bed'):
                        st.write('Warning standing detected')
            # Draw the pose annotation on the image.
            annotated_image = img.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            if len(self.buffer) > 5:
                self.buffer.pop(0)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


webrtc_streamer(key="example",
                video_processor_factory=VideoTransformer,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
