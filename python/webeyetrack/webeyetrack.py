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
from .helpers import relative, relativeT

# Indices of the landmarks that correspond to the eyes and irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_EYE_CENTER = [386, 374]
RIGHT_EYE_CENTER = [159, 145]

HEADPOSE = [4, 152, 236, 33, 287, 57]

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
    
    def compute_depth(self, frame: np.ndarray, iris_center_l: np.ndarray, iris_radius_l: float, iris_center_r: np.ndarray, iris_radius_r: float):
        
        # Depth for the irises
        h,w = frame.shape[:2]
        image_size = (w,h)
        focal_length_pixel = w
        depth_l = estimate_depth(iris_radius_l*2, iris_center_l, focal_length_pixel, np.array(image_size))
        depth_r = estimate_depth(iris_radius_r*2, iris_center_r, focal_length_pixel, np.array(image_size))
        # cv2.putText(frame, f'Depth L: {depth_l:.2f} cm', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f'Depth R: {depth_r:.2f} cm', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return depth_l, depth_r
    
    def process(self, frame: np.ndarray):
        start = time.perf_counter()

        # Package the frame into a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        h, w = frame.shape[:2]

        # Detect the face
        detection_results = self.detector.detect(mp_image)
        points = detection_results.face_landmarks[0]
        
        """
        The gaze function gets an image and face landmarks from mediapipe framework.
        The function draws the gaze direction into the frame.
        """

        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(points[4], frame.shape),  # Nose tip
            relative(points[152], frame.shape),  # Chin
            relative(points[263], frame.shape),  # Left eye left corner
            relative(points[33], frame.shape),  # Right eye right corner
            relative(points[287], frame.shape),  # Left Mouth corner
            relative(points[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points[4], frame.shape),  # Nose tip
            relativeT(points[152], frame.shape),  # Chin
            relativeT(points[263], frame.shape),  # Left eye, left corner
            relativeT(points[33], frame.shape),  # Right eye, right corner
            relativeT(points[287], frame.shape),  # Left Mouth corner
            relativeT(points[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])

        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

        '''
        camera matrix estimation
        '''
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 2d pupil location
        left_pupil = relative(points[468], frame.shape)
        right_pupil = relative(points[473], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

        if transformation is not None:  # if estimateAffine3D seceded
            # project left pupils image point into 3d world point 
            left_pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            right_pupil_world_cord = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            L = Eye_ball_center_left + (left_pupil_world_cord - Eye_ball_center_left) * 10
            R = Eye_ball_center_right + (right_pupil_world_cord - Eye_ball_center_right) * 10

            # Project a 3D gaze direction onto the image plane.
            (left_eye_pupil2D, _) = cv2.projectPoints((int(L[0]), int(L[1]), int(L[2])), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (right_eye_pupil2D, _) = cv2.projectPoints((int(R[0]), int(R[1]), int(R[2])), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)

            # project 3D head pose into the image plane
            (left_head_pose, _) = cv2.projectPoints((int(left_pupil_world_cord[0]), int(left_pupil_world_cord[1]), int(70)),
                                            rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
            (right_head_pose, _) = cv2.projectPoints((int(right_pupil_world_cord[0]), int(right_pupil_world_cord[1]), int(70)),
                                            rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)

            # correct gaze for head rotation
            gaze_left = left_pupil + (left_eye_pupil2D[0][0] - left_pupil) - (left_head_pose[0][0] - left_pupil)
            gaze_right = right_pupil + (right_eye_pupil2D[0][0] - right_pupil) - (right_head_pose[0][0] - right_pupil)

            # Draw gaze line into screen
            L1 = (int(left_pupil[0]), int(left_pupil[1]))
            L2 = (int(gaze_left[0]), int(gaze_left[1]))
            
            R1 = (int(right_pupil[0]), int(right_pupil[1])) 
            R2 = (int(gaze_right[0]), int(gaze_right[1]))
            
            cv2.line(frame, L1, L2, (0, 0, 255), 2)
            cv2.line(frame, R1, R2, (0, 0, 255), 2)
            
            gaze_point =  (int((gaze_left[0] + gaze_right[0]) / 2), int((gaze_left[1] + gaze_right[1]) / 2))
            cv2.circle(frame, gaze_point, 3 , (255, 0, 0), 3)

        # Compute FPS 
        end = time.perf_counter()
        fps = 1 / (end - start)

        return None