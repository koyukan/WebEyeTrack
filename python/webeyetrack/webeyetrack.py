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

from .depth_estimation import from_landmarks_to_depth

# Indices of the landmarks that correspond to the eyes and irises
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_EYE_CENTER = [386, 374]
RIGHT_EYE_CENTER = [159, 145]

HUMAN_IRIS_RADIUS = 11.8  # mm

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
    
    def estimate_depth(self, iris_diameter: float, iris_center: np.ndarray, focal_length_pixel: float, image_size: np.ndarray):
        origin = image_size / 2.0
        y = np.sqrt((origin[0]-iris_center[0])**2 + (origin[1]-iris_center[1])**2)
        x = np.sqrt(focal_length_pixel ** 2 + y ** 2)
        depth_mm = HUMAN_IRIS_RADIUS * x / iris_diameter
        depth_cm = depth_mm / 10
        return depth_cm

    def compute_gaze_direction(self, eye_center, iris_center, depth):
        # Step 1: Compute displacement vector
        dx = iris_center[0] - eye_center[0]
        dy = iris_center[1] - eye_center[1]
        
        # Step 2: Normalize the displacement vector
        magnitude = math.sqrt(dx**2 + dy**2)
        ux = dx / magnitude
        uy = dy / magnitude
        
        # Step 3: Compute pitch and yaw (in radians)
        pitch = math.atan2(uy, math.sqrt(ux**2))
        yaw = math.atan2(ux, 1)  # Assuming uz = 1 for 2D to 3D projection
        
        return pitch, yaw

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
        face_landmarks = detection_results.face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks]
        )
        landmarks = landmarks.T

        h,w = frame.shape[:2]
        image_size = (w,h)

        left_eye_landmarks_id = np.array([33, 133])
        (
            l_depth,
            l_iris_size,
            l_iris_landmarks,
            l_eye_contours
        ) = from_landmarks_to_depth(
            frame,
            landmarks[:, left_eye_landmarks_id],
            image_size,
            is_right_eye=False,
            focal_length = w
        )
        l_depth /= 10

        # Compute depth for the eyes
        # focal_length_pixel = frame.shape[1]
        # depth_l = self.estimate_depth(l_radius, left_eye_center, focal_length_pixel, np.array(image_size))
        # depth_r = self.estimate_depth(r_radius, right_eye_center, focal_length_pixel, np.array(frame.shape[:2]))
        cv2.putText(frame, f'Depth L: {l_depth:.2f} cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


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