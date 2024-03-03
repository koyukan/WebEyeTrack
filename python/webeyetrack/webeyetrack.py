import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
import time

import mediapipe as mp
import cv2

# Face mesh detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

@dataclass
class WebEyeTrackResults():
    detection_results: Any
    draw_image: Optional[np.ndarray]
    fps: float

class WebEyeTrack():

    def __init__(self):
        
        self.face_mesh_detector = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def process(self, frame: np.ndarray, draw_detection: bool = False, draw_informatics: bool = False):

        start = time.perf_counter()
        
        # Detect the face
        detection_results = self.face_mesh_detector(frame)

        # Draw the results
        if draw_detection:
            for face_landmarks in detection_results.multi_face_landmarks:
                # Draw the face mesh annotations on the image.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

        end = time.perf_counter()
        fps = 1 / (end - start)

        # Draw informatics
        if draw_informatics:
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        return WebEyeTrackResults(
            detection_results=detection_results,
            draw_image = frame,
            fps = fps
        )
