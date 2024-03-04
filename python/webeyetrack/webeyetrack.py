import math
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from .utils import estimate_depth, compute_3D_point

# Indices of the landmarks that correspond to the eyes and irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_EYE_CENTER = [386, 374]
RIGHT_EYE_CENTER = [159, 145]

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

        # Get the iris circles
        center_l, l_radius, center_r, r_radius = self.get_iris_circle(
            mesh_points
        )

        # Get the eye contours
        left_eye = mesh_points[LEFT_EYE]
        right_eye = mesh_points[RIGHT_EYE]
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 0, 255), thickness=1)

        # Obtain the center of the eye
        left_eye_center_pts = mesh_points[LEFT_EYE_CENTER]
        left_eye_center = np.mean(left_eye_center_pts, axis=0).astype(int)
        right_eye_center_pts = mesh_points[RIGHT_EYE_CENTER]
        right_eye_center = np.mean(right_eye_center_pts, axis=0).astype(int)

        # Depth
        h,w = frame.shape[:2]
        image_size = (w,h)
        focal_length_pixel = w
        depth_l = estimate_depth(l_radius*1.8, left_eye_center, focal_length_pixel, np.array(image_size))
        depth_r = estimate_depth(r_radius*1.8, right_eye_center, focal_length_pixel, np.array(image_size))
        cv2.putText(frame, f'Depth L: {depth_l:.2f} cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Depth R: {depth_r:.2f} cm', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Construct 3D position of the eye
        eye_3d_l = compute_3D_point(left_eye_center[0], left_eye_center[1], depth_l, h, w)
        eye_3d_r = compute_3D_point(right_eye_center[0], right_eye_center[1], depth_r, h, w)

        # Compute gaze vectors
        # pitch_l, yaw_l = self.compute_gaze_direction(left_eye_center, center_l, depth_l)
        # pitch_r, yaw_r = self.compute_gaze_direction(right_eye_center, center_r, depth_r)

        # Draw the gaze vectors
        # cv2.arrowedLine(frame, tuple(left_eye_center), (int(left_eye_center[0] + 50 * yaw_l), int(left_eye_center[1] - 50 * pitch_l)), (0, 255, 0), 2)
        # cv2.arrowedLine(frame, tuple(right_eye_center), (int(right_eye_center[0] + 50 * yaw_r), int(right_eye_center[1] - 50 * pitch_r)), (0, 255, 0), 2)

        # Compute FPS 
        end = time.perf_counter()
        fps = 1 / (end - start)

        return WebEyeTrackResults(
            detection_results=detection_results,
            left_iris=IrisData(
                center=center_l, 
                radius=l_radius,
            ),
            right_iris=IrisData(
                center=center_r, 
                radius=r_radius
            ),
            fps=fps
        )