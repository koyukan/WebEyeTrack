import math
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
import time
import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import imutils

logger = logging.getLogger("webeyetrack")

from .utils import (
    estimate_depth, 
    compute_3D_point,
    compute_gaze_vector,
    project_3d_pt
)

# Indices of the landmarks that correspond to the eyes and irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_EYE_CENTER = [386, 374]
RIGHT_EYE_CENTER = [159, 145]

HUMAN_EYEBALL_RADIUS = 24 # mm

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
        iris_center_l, iris_radius_l, iris_center_r, iris_radius_r = self.get_iris_circle(
            mesh_points
        )

        # Get the eye contours
        left_eye = mesh_points[LEFT_EYE]
        right_eye = mesh_points[RIGHT_EYE]
        # cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 0, 255), thickness=1)
        # cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 0, 255), thickness=1)

        # Depth for the irises
        h,w = frame.shape[:2]
        image_size = (w,h)
        focal_length_pixel = w
        depth_l = estimate_depth(iris_radius_l*2, iris_center_l, focal_length_pixel, np.array(image_size))
        depth_r = estimate_depth(iris_radius_r*2, iris_center_r, focal_length_pixel, np.array(image_size))
        cv2.putText(frame, f'Depth L: {depth_l:.2f} cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Depth R: {depth_r:.2f} cm', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Compute 3D position of the iris
        iris_3d_l = compute_3D_point(iris_center_l[0], iris_center_l[1], depth_l, h, w)
        iris_3d_r = compute_3D_point(iris_center_r[0], iris_center_r[1], depth_r, h, w)
        logger.debug(f'3D position of the left iris: {iris_3d_l}')

        # Test the 3D position of the iris by project back to 2D
        K = np.array([[focal_length_pixel, 0, w/2], [0, focal_length_pixel, h/2], [0, 0, 1]])
        # projected_iris_l = cv2.projectPoints(iris_3d_l, np.eye(3), np.zeros(3), K, np.zeros(5))
        # projected_iris_r = cv2.projectPoints(iris_3d_r, np.eye(3), np.zeros(3), K, np.zeros(5))
        projected_iris_l = project_3d_pt(iris_3d_l, K)
        # import pdb; pdb.set_trace()
        cv2.circle(frame, (int(projected_iris_l[0]), int(projected_iris_l[1])), 1, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(projected_iris_r[0][0][0].astype(int)), 1, (0, 0, 255), -1)

        # Compute the 3D center of the eye-ball
        eye_center_pts_l = mesh_points[LEFT_EYE_CENTER]
        left_eye_center = np.mean(eye_center_pts_l, axis=0).astype(int)
        eye_center_pts_r = mesh_points[RIGHT_EYE_CENTER]
        right_eye_center = np.mean(eye_center_pts_r, axis=0).astype(int)

        # Draw the eye centers
        # cv2.circle(frame, tuple(iris_center_l), 1, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(left_eye_center), 1, (0, 255, 0), -1)

        # Crop the eye region 
        eye_l = frame[left_eye_center[1]-50:left_eye_center[1]+50, left_eye_center[0]-50:left_eye_center[0]+50]
        cv2.imshow('eye_l', imutils.resize(eye_l, width=300))
        cv2.waitKey(0)

        # Compute 3D position of the eye center
        eye_3d_l = compute_3D_point(left_eye_center[0], left_eye_center[1], depth_l+(HUMAN_EYEBALL_RADIUS/10), h, w)
        eye_3d_r = compute_3D_point(right_eye_center[0], right_eye_center[1], depth_r+(HUMAN_EYEBALL_RADIUS/10), h, w)


        # Compute gaze vectors
        gaze_vector_l = compute_gaze_vector(iris_3d_l, eye_3d_l)
        gaze_vector_r = compute_gaze_vector(iris_3d_r, eye_3d_r)

        # Draw the gaze vectors
        # cv2.arrowedLine(frame, tuple(left_eye_center), (int(left_eye_center[0] + 50 * gaze_vector_l[0]), int(left_eye_center[1] - 50 * gaze_vector_l[1])), (0, 255, 0), 1)
        eye_l = frame[left_eye_center[1]-50:left_eye_center[1]+50, left_eye_center[0]-50:left_eye_center[0]+50]
        cv2.imshow('eye_l', imutils.resize(eye_l, width=300))
        cv2.waitKey(0)

        # Draw the gaze vectors
        # cv2.arrowedLine(frame, tuple(left_eye_center), (int(left_eye_center[0] + 50 * yaw_l), int(left_eye_center[1] - 50 * pitch_l)), (0, 255, 0), 2)
        # cv2.arrowedLine(frame, tuple(right_eye_center), (int(right_eye_center[0] + 50 * yaw_r), int(right_eye_center[1] - 50 * pitch_r)), (0, 255, 0), 2)

        # Compute FPS 
        end = time.perf_counter()
        fps = 1 / (end - start)

        return WebEyeTrackResults(
            detection_results=detection_results,
            left_iris=IrisData(
                center=iris_center_l, 
                radius=iris_radius_l,
            ),
            right_iris=IrisData(
                center=iris_center_r, 
                radius=iris_radius_r
            ),
            fps=fps
        )