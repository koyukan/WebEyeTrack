import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Face mesh detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

@dataclass
class WebEyeTrackResults():
    detection_results: Any
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

    def process(self, frame: np.ndarray):
        start = time.perf_counter()

        # Package the frame into a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect the face
        detection_results = self.detector.detect(mp_image)
        
        end = time.perf_counter()
        fps = 1 / (end - start)

        return WebEyeTrackResults(
            detection_results=detection_results,
            fps=fps
        )