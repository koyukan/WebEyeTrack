import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Indices of the landmarks that correspond to the eyes and irises
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

@dataclass
class IrisData():
    center: np.ndarray
    radius: float


@dataclass
class WebEyeTrackResults():
    detection_results: Any
    left_iris: IrisData
    right_iris: IrisData
    fps: float


class WebEyeTrack():

    def __init__(self, model_path: Union[str, pathlib.Path]):
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_iris_circle(self, landmarks: np.ndarray):

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(landmarks[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(landmarks[RIGHT_IRIS])

        center_l = np.array([l_cx, l_cy], dtype=np.int32)
        center_r = np.array([r_cx, r_cy], dtype=np.int32)
        return center_l, l_radius, center_r, r_radius

    def process(self, frame: np.ndarray):
        start = time.perf_counter()

        # Package the frame into a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        h, w = frame.shape[:2]
        
        # Detect the face
        detection_results = self.detector.detect(mp_image)
        mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in detection_results.face_landmarks[0]])

        center_l, l_radius, center_r, r_radius = self.get_iris_circle(
            mesh_points
        )
        
        end = time.perf_counter()
        fps = 1 / (end - start)

        return WebEyeTrackResults(
            detection_results=detection_results,
            left_iris=IrisData(
                center=center_l, 
                radius=l_radius
            ),
            right_iris=IrisData(
                center=center_r, 
                radius=r_radius
            ),
            fps=fps
        )