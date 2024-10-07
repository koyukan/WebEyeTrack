import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import math

def draw_axis(img, yaw, pitch, roll=0, tdx=None, tdy=None, size = 100):

    pitch = -(pitch * np.pi / 180)
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),3)

    return img

def draw_gaze_from_vector(img, face_origin_2d, gaze_direction_3d, scale=100, color=(0, 0, 255)):
    """
    Draws a 2D gaze direction based on a 3D unit gaze vector projected onto a 2D plane.
    
    Arguments:
    img -- the image where the gaze will be drawn
    face_origin_2d -- the 2D position of the face origin (where the gaze starts)
    gaze_direction_3d -- the 3D gaze direction vector
    scale -- the scaling factor for the length of the arrow
    color -- the color of the gaze direction line
    """
    # Normalize the 3D gaze direction vector
    gaze_direction_3d = gaze_direction_3d / np.linalg.norm(gaze_direction_3d)

    # Project the 3D gaze vector onto the 2D plane (ignore Z-axis)
    gaze_target_2d = face_origin_2d + scale * np.array([gaze_direction_3d[0], gaze_direction_3d[1]])

    # Draw the gaze direction as an arrowed line
    cv2.arrowedLine(img, 
                    (int(face_origin_2d[0]), int(face_origin_2d[1])), 
                    (int(gaze_target_2d[0]), int(gaze_target_2d[1])), 
                    color, 3, tipLength=0.3)

    return img

def draw_axis_from_rotation_matrix(img, R, tdx=None, tdy=None, size=100):
    """
    Draws the transformed 3D axes using the provided rotation matrix `R`.
    """
    # Define the 3D axes (X, Y, Z)
    x_axis = np.array([1, 0, 0])  # X-axis (Red)
    y_axis = np.array([0, 1, 0])  # Y-axis (Green)
    z_axis = np.array([0, 0, 1])  # Z-axis (Blue)
    
    # Apply the rotation matrix to the axes
    x_rot = np.dot(R, x_axis)
    y_rot = np.dot(R, y_axis)
    z_rot = np.dot(R, z_axis)
    
    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width // 2
        tdy = height // 2

    # Project the transformed axes onto the image (2D)
    x1 = size * x_rot[0] + tdx
    y1 = size * x_rot[1] + tdy
    x2 = size * y_rot[0] + tdx
    y2 = size * y_rot[1] + tdy
    x3 = size * z_rot[0] + tdx
    y3 = size * z_rot[1] + tdy

    # Draw the axes
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-axis (Red)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-axis (Green)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)  # Z-axis (Blue)

    return img

def draw_gaze_origin_heatmap(image, heatmap, alpha=0.5, cmap='jet'):
    """
    Overlay a semi-transparent heatmap on an image.

    Parameters:
    - image: numpy array of shape (h, w, 3)
    - heatmap: numpy array of shape (h, w)
    - alpha: float, transparency level of the heatmap
    - cmap: str, colormap to use for the heatmap

    Returns:
    - overlay: numpy array of the overlaid image
    """
    # Normalize the heatmap to be between 0 and 1
    heatmap_normalized = Normalize()(heatmap)
    
    # Create a color map
    colormap = plt.get_cmap(cmap)
    
    # Apply the colormap to the heatmap
    heatmap_colored = colormap(heatmap_normalized)
    
    # Remove the alpha channel from the colormap output
    heatmap_colored = heatmap_colored[:, :, :3]
    
    # Overlay the heatmap on the image
    overlay = image * (1 - alpha) + heatmap_colored * alpha
    
    # Clip the values to be in the valid range [0, 1]
    overlay = np.clip(overlay, 0, 1)

    return overlay

def draw_gaze_depth_map(image, depth_map, alpha=0.5, cmap='jet'):
    # Normalize the heatmap to be between 0 and 1
    heatmap_normalized = Normalize()(depth_map)
    
    # Create a color map
    colormap = plt.get_cmap(cmap)
    
    # Apply the colormap to the heatmap
    heatmap_colored = colormap(heatmap_normalized)
    
    # Remove the alpha channel from the colormap output
    heatmap_colored = heatmap_colored[:, :, :3]
    
    # Overlay the heatmap on the image
    overlay = image * (1 - alpha) + heatmap_colored * alpha
    
    # Clip the values to be in the valid range [0, 1]
    overlay = np.clip(overlay, 0, 1)

    return overlay

def draw_gaze_origin(image, gaze_origin, color=(255, 0, 0)):
    # Draw gaze origin
    draw_image = image.copy()
    x, y = gaze_origin
    cv2.circle(draw_image, (int(x), int(y)), 10, color, -1)

    return draw_image

def draw_gaze_direction(image, gaze_origin, gaze_dst, color=(255, 0, 0)):
    # Draw gaze direction
    draw_image = image.copy()
    x, y = gaze_origin
    dx, dy = gaze_dst
    cv2.arrowedLine(draw_image, (int(x), int(y)), (int(dx), int(dy)), color, 2)

    return draw_image

def draw_pog(img, pog, color=(255,0,0)):
    # Draw point of gaze (POG)
    x, y = pog
    cv2.circle(img, (int(x), int(y)), 10, color, -1)
    return img