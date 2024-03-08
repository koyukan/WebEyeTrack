import math
import numpy as np
from scipy.spatial.transform import Rotation as R

HUMAN_IRIS_RADIUS = 11.8  # mm

def estimate_depth(iris_diameter: float, iris_center: np.ndarray, focal_length_pixel: float, image_size: np.ndarray):
    origin = image_size / 2.0
    y = np.sqrt((origin[0]-iris_center[0])**2 + (origin[1]-iris_center[1])**2)
    x = np.sqrt(focal_length_pixel ** 2 + y ** 2)
    depth_mm = HUMAN_IRIS_RADIUS * x / iris_diameter
    depth_cm = depth_mm / 10
    return depth_cm


def compute_3D_point(x, y, Z, H, W):
    """
    Compute the 3D point in the camera coordinate system from an image coordinate and depth.

    Parameters:
    - x, y: The image coordinates (pixels)
    - Z: The depth value (distance along the camera's viewing axis)
    - f_x, f_y: The camera's focal lengths along the X and Y axes (pixels)
    - c_x, c_y: The optical center of the camera (pixels)

    Returns:
    A tuple (X, Y, Z) representing the 3D point in the camera coordinate system.
    """
    # 
    # fy = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    # fx = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    fy = 0.5 * W / np.tan(0.5 * 50 * np.pi / 180.0)
    fx = 0.5 * W / np.tan(0.5 * 50 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H

    # Normalize the 2D coordinates
    x_prime = (x - cx) / fx
    y_prime = (y - cy) / fy

    # Apply the depth to get the 3D point
    X = x_prime * Z
    Y = y_prime * Z

    return np.array([X, Y, Z])


def compute_gaze_vector(iris_t, eyeball_t):

    # Compute the 3D rotation vector from the eyeball to the iris
    gaze_vector = iris_t - eyeball_t
    gaze_vector /= np.linalg.norm(gaze_vector)

    # Convert to pitch and yaw
    pitch = math.asin(-gaze_vector[1])
    yaw = math.atan2(-gaze_vector[0], -gaze_vector[2])

    # Convert to rotation vector
    rotation = R.from_euler('xyz', [0, yaw, pitch], degrees=False)
    return rotation

def project_3d_pt(pt, K):
    """
    Project a 3D point to 2D using the camera matrix K.

    Parameters:
    - pt: The 3D point (X, Y, Z)
    - K: The camera matrix

    Returns:
    A tuple (x, y) representing the 2D point in the image.
    """
    pt = pt.reshape((3, 1))
    pt = np.dot(K, pt)
    pt = pt / pt[2]
    return (pt[0], pt[1])

def compute_head_pose():
    ...