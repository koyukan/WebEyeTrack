from dataclasses import dataclass
import numpy as np
import pathlib

@dataclass
class Annotations:
    original_img_size: np.ndarray # (3,)
    pog_px: np.ndarray # (2,)
    facial_landmarks: np.ndarray # (5, N)
    facial_landmarks_2d: np.ndarray # (2, N)
    facial_rt: np.ndarray # (4, 4)
    face_blendshapes: np.ndarray # (N,)
    face_bbox: np.ndarray # (4,)
    head_pose_3d: np.ndarray # (6,), rotation matrix
    face_origin_3d: np.ndarray # (3,)
    face_origin_2d: np.ndarray # (2,)
    gaze_target_3d: np.ndarray # (3,)
    gaze_target_2d: np.ndarray # (2,)
    gaze_direction_3d: np.ndarray # (3,)
    # which_eye: str # (left, right)

@dataclass
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    camera_retval: float
    camera_rvecs: np.ndarray
    camera_tvecs: np.ndarray
    monitor_rvecs: np.ndarray
    monitor_tvecs: np.ndarray
    monitor_height_mm: float
    monitor_height_px: float
    monitor_width_mm: float
    monitor_width_px: float

@dataclass
class Sample:
    participant_id: str
    image_fp: pathlib.Path
    annotations: Annotations


@dataclass
class EyeResult:
    is_closed: bool
    origin: np.ndarray # X, Y, Z
    origin_2d: np.ndarray # u, v
    direction: np.ndarray # X, Y, Z
    pog_px: np.ndarray
    pog_mm: np.ndarray

@dataclass
class FLGEResult:
    facial_landmarks: np.ndarray
    face_rt: np.ndarray
    face_blendshapes: np.ndarray
    face_origin: np.ndarray # X, Y, Z
    face_origin_2d: np.ndarray # X, Y
    face_gaze: np.ndarray # X, Y, Z
    left: EyeResult
    right: EyeResult
    pog_px: np.ndarray
    pog_mm: np.ndarray
    duration: float # seconds