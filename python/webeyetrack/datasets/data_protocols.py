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
    face_blendshapes: list # (N, 3)
    face_bbox: np.ndarray # (4,)
    head_pose_3d: np.ndarray # (6,), rotation matrix
    face_origin_3d: np.ndarray # (3,)
    face_origin_2d: np.ndarray # (2,)
    gaze_target_3d: np.ndarray # (3,)
    gaze_target_2d: np.ndarray # (2,)
    gaze_direction_3d: np.ndarray # (3,)
    which_eye: str # (left, right)

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