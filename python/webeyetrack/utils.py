import math
import numpy as np

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
    - H, W: The height and width of the image (pixels)

    Returns:
    A tuple (X, Y, Z) representing the 3D point in the camera coordinate system.
    """
    # 
    fy = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    fx = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H

    # Normalize the 2D coordinates
    x_prime = (x - cx) / fx
    y_prime = (y - cy) / fy

    # Apply the depth to get the 3D point
    X = x_prime * Z
    Y = y_prime * Z

    return np.array([X, Y, Z])

def compute_gaze_direction(eye_center, iris_center, depth):
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