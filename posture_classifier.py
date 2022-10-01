import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import _pickle as cPickle

from pose_encoder import FullBodyPoseEmbedder

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# encoder = FullBodyPoseEmbedder()
# pos_data = pd.DataFrame([])
# for dir in os.listdir('/content/drive/MyDrive/Takeout/Takeout/Google Fotos/training_set'):
#     print(dir)
#     for image in os.listdir(os.path.join('/content/drive/MyDrive/Takeout/Takeout/Google Fotos/training_set', dir)):
#         image = cv2.imread(os.path.join('/content/drive/MyDrive/Takeout/Takeout/Google Fotos/training_set', dir, image))
#         with mp_pose.Pose(
#                 static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
#             # results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             # plt.imshow(np.array(image))
#             results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             if results is None:
#                 continue
#             if results.pose_landmarks is None:
#                 continue
#             pose_landmarks = np.array([[lmk.x * 1, lmk.y * 1, lmk.z * 1]
#                                        for lmk in results.pose_landmarks.landmark], dtype=np.float32)
#             encoder = FullBodyPoseEmbedder()
#             embedding = encoder(pose_landmarks)
#
#             pos_data = pos_data.append({'cat': dir, 'name': image, 'data': embedding.flatten()}, ignore_index=True)


def detect_pose(results):
    if results.pose_landmarks is None:
        return None
    pose_landmarks = np.array([[lmk.x * 1, lmk.y * 1, lmk.z * 1]
                               for lmk in results.pose_landmarks.landmark], dtype=np.float32)

    pos_data = pd.DataFrame([])
    encoder = FullBodyPoseEmbedder()
    embedding = encoder(pose_landmarks)
    x_cols = [str(i) for i in range(len(embedding.flatten()))]
    pos_data = pos_data.append({'data': embedding.flatten()}, ignore_index=True)
    pos_data[x_cols] = pd.DataFrame(pos_data['data'].tolist(), index=pos_data.index)

    with open('random_forest_classifer.pkl', 'rb') as fid:
        gnb_loaded = cPickle.load(fid)

    res = gnb_loaded.predict(pos_data[x_cols])

    dict = {0: 'bed', 1: 'floor', 2: 'sitting', 3: 'standing'}
    return dict[res.argmax()]

# img = cv2.imread('20221001_195628.jpg')
#
# with mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as pose:
#     results = pose.process(img)
#     detected_pose = detect_pose(results)
#     # Draw the pose annotation on the image.
#     annotated_image = img.copy()
#     mp_drawing.draw_landmarks(
#         annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
# print(detect_pose(results))
