import os
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

from .translation_kalman_filter import TranslationKalmanFilter
# from .pose_kalman_filter import PoseKalmanFilter
from .vis import draw_landmarks_on_image
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

# HEADPOSE = [4, 152, 263, 33, 287, 57]
HEADPOSE = range(0, 468)

HUMAN_EYEBALL_RADIUS = 24 # mm

CWD = pathlib.Path(os.path.abspath(__file__)).parent

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


def compute_pog(eye_center, eye_pupil2D, screen_origin, screen_normal):
    """
    Compute the PoG (Point of Gaze) by intersecting the gaze ray with the screen plane.
    screen_origin is the origin of the screen plane (e.g., [0, 0, 0]).
    screen_normal is the normal vector of the screen plane (e.g., [0, 0, 1]).
    """
    # Convert points to numpy arrays
    o = np.array(eye_center)
    r = np.array(eye_pupil2D) - o
    r /= np.linalg.norm(r)

    numer = np.dot(screen_normal, screen_origin - o)
    denom = np.dot(screen_normal, r) + 1e-6

    lambda_distance = numer / denom
    # print(lambda_distance)
    pog = o + lambda_distance * r
    pog = pog[:2] / 10

    return pog


def transform_to_camera_coordinates(world_points, rvec, tvec):
    """
    Transforms points from world coordinates to camera coordinates.

    Args:
    - world_points: Nx3 array of points in world coordinates.
    - rvec: Rotation vector.
    - tvec: Translation vector.

    Returns:
    - camera_points: Nx3 array of points in camera coordinates.
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create the 4x4 transformation matrix
    RT = np.hstack((R, tvec))
    RT = np.vstack((RT, [0, 0, 0, 1]))
    
    # Convert world points to homogeneous coordinates (Nx4)
    ones = np.ones((world_points.shape[0], 1))
    world_points_hom = np.hstack((world_points, ones))
    
    # Apply the transformation matrix
    camera_points_hom = RT @ world_points_hom.T
    
    # Convert back to 3xN and return
    camera_points = camera_points_hom[:3].T
    return camera_points


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
        self.translation_filter = TranslationKalmanFilter()
        self.canonical_face = np.load(CWD / 'assets' / "canonical_face_model.npy")

        # Center the canonical face model around the nose
        self.canonical_face -= self.canonical_face[4]

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
        if detection_results is None or len(detection_results.face_landmarks) == 0: return frame
        points = detection_results.face_landmarks[0]
        detected_canonical_face = points[:468]
        frame = draw_landmarks_on_image(frame, detection_results)

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
            relative(points[x], frame.shape) for x in HEADPOSE
        ], dtype="double")
        # import pdb; pdb.set_trace()

        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points[x], frame.shape) for x in HEADPOSE
        ], dtype="double")

        # 3D model points.
        # model_points = np.array([
        #     (0.0, 0.0, 0.0),  # Nose tip
        #     (0, -63.6, -12.5),  # Chin
        #     (-43.3, 32.7, -26),  # Left eye, left corner
        #     (43.3, 32.7, -26),  # Right eye, right corner
        #     (-28.9, -28.9, -24.1),  # Left Mouth corner
        #     (28.9, -28.9, -24.1)  # Right mouth corner
        # ])
        model_points = self.canonical_face[HEADPOSE]
        # model_points[:, 0] *= -1
        # model_points = np.array([
        #     points[x].z for x in HEADPOSE
        # ])
        RATIO = np.array([9.52, 8.57, 5.87])
        # model_points = model_points * RATIO
        # import pdb; pdb.set_trace()

        '''
        3D model eye points
        The center of the eye ball
        '''
        # Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        # Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.
        # X = 29.05
        # Y = 29.7
        # Z = -39.5
        # Eye_ball_center_right = np.array([[-X], [Y], [Z]])
        # Eye_ball_center_left = np.array([[X], [Y], [Z]])  # the center of the left eyeball as a vector.
        X = -3.01
        Y = 3.1875
        Z = -5.90
        Eye_ball_center_left = np.array([[X], [Y], [Z]])  # the center of the left eyeball as a vector.
        Eye_ball_center_right = np.array([[-X], [Y], [Z]])
        # Eye_ball_center_right = np.array([[-3.09], [4.0875], [-7.90]])
        # Eye_ball_center_left = np.array([[3.09], [4.0875], [-7.50]])  # the center of the left eyeball as a vector.

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
        (success, rvec, tvec, inliners) = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            logger.error("Failed to solve PnP")
            return frame, (-1, -1)
        
        # Kalman filter for pose
        # tvec, _ = self.translation_filter.process(tvec)
        # rvec, tvec, _ = self.pose_kalman_filter.process(rvec, tvec)

        # 2d pupil location
        left_pupil = relative(points[468], frame.shape)
        right_pupil = relative(points[473], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

        if transformation is not None:  # if estimateAffine3D seceded

            # project the 3D head pose into the image plane
            (nose_start_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, int(0))]), rvec, tvec, camera_matrix, dist_coeffs)
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, int(100.0))]), rvec, tvec, camera_matrix, dist_coeffs)

            # project left pupils image point into 3d world point 
            left_pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            right_pupil_world_cord = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            length = 10
            L = Eye_ball_center_left + (left_pupil_world_cord - Eye_ball_center_left) * length
            R = Eye_ball_center_right + (right_pupil_world_cord - Eye_ball_center_right) * length

            # Draw the eye ball center
            (left_eye_center, _) = cv2.projectPoints(Eye_ball_center_left, rvec, tvec, camera_matrix, dist_coeffs)
            (right_eye_center, _) = cv2.projectPoints(Eye_ball_center_right, rvec, tvec, camera_matrix, dist_coeffs)
            cv2.circle(frame, (int(left_eye_center[0][0][0]), int(left_eye_center[0][0][1])), 3, (0, 255, 0), 3)
            cv2.circle(frame, (int(right_eye_center[0][0][0]), int(right_eye_center[0][0][1])), 3, (0, 255, 0), 3)

            # Project a 3D gaze direction onto the image plane.
            (left_eye_pupil2D, _) = cv2.projectPoints((int(L[0]), int(L[1]), int(L[2])), rvec,
                                                tvec, camera_matrix, dist_coeffs)
            (right_eye_pupil2D, _) = cv2.projectPoints((int(R[0]), int(R[1]), int(R[2])), rvec,
                                                tvec, camera_matrix, dist_coeffs)

            # project 3D head pose into the image plane
            (left_head_pose, _) = cv2.projectPoints((int(left_pupil_world_cord[0]), int(left_pupil_world_cord[1]), int(70)),
                                            rvec,
                                            tvec, camera_matrix, dist_coeffs)
            (right_head_pose, _) = cv2.projectPoints((int(right_pupil_world_cord[0]), int(right_pupil_world_cord[1]), int(70)),
                                            rvec,
                                            tvec, camera_matrix, dist_coeffs)

            # correct gaze for head rotation
            gaze_left = left_pupil + (left_eye_pupil2D[0][0] - left_pupil) - (left_head_pose[0][0] - left_pupil)
            gaze_right = right_pupil + (right_eye_pupil2D[0][0] - right_pupil) - (right_head_pose[0][0] - right_pupil)

            # Draw gaze line into screen
            L1 = (int(left_pupil[0]), int(left_pupil[1]))
            L2 = (int(gaze_left[0]), int(gaze_left[1]))
            
            R1 = (int(right_pupil[0]), int(right_pupil[1])) 
            R2 = (int(gaze_right[0]), int(gaze_right[1]))

            # H1 = (int(nose_start_point2D[0][0][0]), int(nose_start_point2D[0][0][1]))
            # H2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # cv2.line(frame, H1, H2, (0, 0, 255), 2)
            cv2.line(frame, L1, L2, (255, 0, 0), 2)
            cv2.line(frame, R1, R2, (255, 0, 0), 2)
            
            gaze_point =  (int((gaze_left[0] + gaze_right[0]) / 2), int((gaze_left[1] + gaze_right[1]) / 2))
            # cv2.circle(frame, gaze_point, 3 , (255, 0, 0), 3)

        # Compute the PoG
        # Transform the eye ball center to camera coordinates
        # import pdb; pdb.set_trace()
        eye_ball_center_left_camera = transform_to_camera_coordinates(Eye_ball_center_left.reshape((-1, 3)), rvec, tvec)
        left_pupil_camera = transform_to_camera_coordinates(left_pupil_world_cord.reshape((-1, 3)), rvec, tvec)
        pog = compute_pog(eye_ball_center_left_camera.flatten(), left_pupil_camera.flatten(), np.array([1, 0, 0]), np.array([0, 0, 1]))
        print(pog)

        # Compute FPS 
        end = time.perf_counter()
        fps = 1 / (end - start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, pog