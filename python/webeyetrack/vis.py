from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import math
import imutils
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from .data_protocols import GazeResult, EyeResult
from .model_based import vector_to_pitch_yaw, rotation_matrix_to_euler_angles
from .constants import *

EYE_IMAGE_WIDTH = 400

#####################################################################################################
# POG
######################################################################################################

def plot_2d_dist(x_vals, y_vals, title):
    # 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, range=[[-0.5, 0.5],[-0.5, 0.5]], bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(
        heatmap.T,
        extent=extent,
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.colorbar(label='Frequency')
    xlabel = 'X (Normalized)'
    ylabel = 'Y (Normalized)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    return fig

def plot_pog_errors(gt_x, gt_y, pred_x, pred_y):

    # Also create a plot showing the errows between predicted and ground truth PoGs
    # Scatter plot with error lines
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Plot ground truth points
    ax.scatter(gt_x, gt_y, color='lime', label='Ground Truth', s=10, alpha=0.8)

    # Plot predicted points
    ax.scatter(pred_x, pred_y, color='red', label='Predicted', s=10, alpha=0.8)

    # Plot error vectors (lines between GT and predicted)
    for (gx, gy, px, py) in zip(gt_x, gt_y, pred_x, pred_y):
        ax.plot([gx, px], [gy, py], color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (Normalized)')
    ax.set_ylabel('Y (Normalized)')
    ax.set_title('Gaze Prediction Error Vectors')
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig

class TimeSeriesOscilloscope:

    def __init__(self, name: str, min_value: float, max_value: float, num_points: int, px_height: int = 400, pxs_per_point: int = 10):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.num_points = num_points
        self.px_height = px_height
        self.pxs_per_point = pxs_per_point

        # self.img = np.zeros((num_points, px_height, 3), dtype=np.uint8)
        self.img = np.zeros((px_height, num_points * self.pxs_per_point, 3), dtype=np.uint8)

    def update(self, value: float) -> np.ndarray:
        norm_value = (value - self.min_value) / (self.max_value - self.min_value)
        norm_value = np.clip(norm_value, 0, 1)
        px_value = int(norm_value * self.px_height)

        # Shift the image by pxs_per_point along the width
        self.img[:, :-self.pxs_per_point] = self.img[:, self.pxs_per_point:]

        # Draw the new value
        self.img[:, -self.pxs_per_point:] = 0
        self.img[-px_value:, -self.pxs_per_point:] = 255

        return self.img

def draw_landmarks_simple(draw_frame, landmarks):
    for pt in landmarks:
        cv2.circle(draw_frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

def draw_landmarks_on_image(rgb_image, detection_result):
    if type(detection_result) == dict:
        face_landmarks_list = detection_result['face_landmarks']
    else:
        face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image
    
def pad_and_concat_images(image1, image2):
    # Get dimensions of both images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Determine the new dimensions (max height and width)
    new_height = max(h1, h2)
    new_width = max(w1, w2)

    # Create new black images with the target dimensions
    padded_img1 = np.zeros((new_height, new_width, 3), dtype=image1.dtype)
    padded_img2 = np.zeros((new_height, new_width, 3), dtype=image2.dtype)

    # Copy original images into the padded ones
    padded_img1[:h1, :w1] = image1
    padded_img2[:h2, :w2] = image2

    # Horizontally concatenate the two padded images
    hconcat_image = np.hstack((padded_img1, padded_img2))

    return hconcat_image

def landmark_gaze_render(frame: np.ndarray, result: GazeResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]

    intrinsics = np.array([
        [frame.shape[1], 0, frame.shape[1] / 2],
        [0, frame.shape[0], frame.shape[0] / 2],
        [0, 0, 1]
    ]).astype(np.float32)

    # Compute the bbox by using the edges of the each eyes
    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_px = facial_landmarks[LEFT_EYELID_LANDMARKS, :2] * np.array([width, height])
    left_2d_iris_px = facial_landmarks[LEFT_IRIS_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyearea_total_px = facial_landmarks[LEFT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_total_px = facial_landmarks[LEFT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])
    
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_px = facial_landmarks[RIGHT_EYELID_LANDMARKS, :2] * np.array([width, height])
    right_2d_iris_px = facial_landmarks[RIGHT_IRIS_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyearea_total_px = facial_landmarks[RIGHT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_total_px = facial_landmarks[RIGHT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])

    left_landmarks = [
        left_2d_eye_px,
        left_2d_eyelid_px,
        left_2d_eyearea_total_px,
        left_2d_eyelid_total_px,
    ]

    right_landmarks = [
        right_2d_eye_px,
        right_2d_eyelid_px,
        right_2d_eyearea_total_px,
        right_2d_eyelid_total_px,
    ]

    eye_images = {}
    for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        eye_result = result.left if i == 'left' else result.right
        is_closed = eye_result.is_closed

        # Determine if the eye is closed by the ratio of the height based on the width
        # if eyelid_height / eyelid_width < 0.05:
        #     is_closed = True

        if width == 0 or height == 0:
            continue

        # Crop the eye
        eye_image = frame[
            int(centroid[1] - height/2):int(centroid[1] + height/2),
            int(centroid[0] - width/2):int(centroid[0] + width/2)
        ]

        # Create eye image
        original_height, original_width = eye_image.shape[:2]
        new_width, new_height = EYE_IMAGE_WIDTH, int(EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO)
        eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        eye_images[i] = eye_image

        # Draw the outline of the eyearea
        shifted_eyearea_px = eyearea - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        prior_px = None
        for px in shifted_eyearea_px:
            resized_px = px * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (0, 255, 0), 1)
            prior_px = resized_px
        # Draw the last line to close the loop
        resized_first_px = shifted_eyearea_px[0] * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (0, 255, 0), 1)

        # Draw the outline of the eyelid
        shifted_eyelid_px = eyelid_total - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        prior_px = None
        for px in shifted_eyelid_px:
            resized_px = px * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
            prior_px = resized_px
        
        # Draw the last line to close the loop
        resized_first_px = shifted_eyelid_px[0] * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        for iris_px_pt in shifted_iris_px:
            resized_iris_px = iris_px_pt * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
            cv2.circle(eye_image, tuple(resized_iris_px.astype(int)), 3, (0, 0, 255), -1)

        # Draw the centroid of the eyeball
        cv2.circle(eye_image, (int(new_width/2), int(new_height/2)), 3, (255, 0, 0), -1)
        
        # Compute the line between the iris center and the centroid
        new_shifted_iris_px_center = shifted_iris_px[0] * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, (int(new_width/2), int(new_height/2)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 0), 2)

        # If available, visualize the headpose-corrected iris center
        if 'headpose_corrected_eye_center' in eye_result.meta_data:
            headpose_corrected_eye_center = eye_result.meta_data['headpose_corrected_eye_center']
            if headpose_corrected_eye_center is not None:
                new_headpose_corrected_eye_center = headpose_corrected_eye_center * np.array([EYE_IMAGE_WIDTH/original_width, EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO/original_height])
                cv2.circle(eye_image, tuple(new_headpose_corrected_eye_center.astype(int)), 3, (0, 255, 255), -1)

                # Draw a line between the headpose corrected iris center and the iris center
                cv2.line(eye_image, tuple(new_headpose_corrected_eye_center.astype(int)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 255), 2)
  
        if is_closed:
            cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Convert 3D to pitch and yaw
        iris_center = iris_px[0]
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        frame = draw_axis(frame, pitch, yaw, 0, int(iris_center[0]), int(iris_center[1]), 100)

    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the headpose on the frame
    # headrot = result.face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = result.face_origin_2d
    # frame = draw_axis(frame, pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # Concatenate the images
    right_eye_image = eye_images['right']
    left_eye_image = eye_images['left']
    eyes_combined = cv2.hconcat([right_eye_image, left_eye_image])

    # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
    eyes_combined_resized = cv2.resize(eyes_combined, (frame.shape[1], eyes_combined.shape[0]))

    # Concatenate the combined eyes image vertically with the frame
    return cv2.vconcat([frame, eyes_combined_resized])

def blendshape_gaze_render(frame: np.ndarray, result: GazeResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]

    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([width, height])

    for i, eye in {'left': left_2d_eye_px, 'right': right_2d_eye_px}.items():
        centroid = np.mean(eye, axis=0)
        eye_result = result.left if i == 'left' else result.right 

        # Apply a correct R to the gaze direction
        # R = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, -1]
        # ])
        # rotvec = np.array([90, 0, 0], dtype=np.float32) # in degree
        rotvec = np.array([0, 90, 0], dtype=np.float32) # in degree
        rotvec = np.radians(rotvec)
        R = cv2.Rodrigues(rotvec)[0]
        corrected_gaze_direction = np.dot(R, eye_result.direction) * np.array([1, 1, 1])

        # if i == 'right':
        #     pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        #     frame = draw_axis(frame, pitch, yaw, 0, int(centroid[0]), int(centroid[1]), 100)
        #     cv2.putText(frame, f"{i}", (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # else:
        pitch, yaw = vector_to_pitch_yaw(corrected_gaze_direction)
        frame = draw_axis(frame, -yaw, pitch, 0, int(centroid[0]), int(centroid[1]), 100)
        # cv2.putText(frame, f"{i}", (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def draw_axis(img, pitch, yaw, roll=0, tdx=None, tdy=None, size=100, color=(255,255,255), show_xy=False):
    """
    Draws the 3D axes based on the given pitch, yaw, and roll. The Z-axis is drawn
    pointing towards the negative Z direction.
    
    Arguments:
    img -- the image to draw the axes on
    pitch -- pitch angle in degrees
    yaw -- yaw angle in degrees
    roll -- roll angle in degrees
    tdx -- x translation (optional)
    tdy -- y translation (optional)
    size -- the length of the axis to be drawn
    """
    pitch = (pitch * np.pi / 180)
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
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

    # Z-Axis (negative Z) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (math.cos(yaw) * math.sin(pitch)) + tdy  # Note the change here for negative Z

    # Draw the axes with appropriate colors
    try:
        if show_xy:
            cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 0), 3, tipLength=0.2)  # X-axis (Black)
            cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (color[0]/2, color[1]/2, color[2]/2), 3, tipLength=0.2)  # Y-axis (Gray)
        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), color, 3, tipLength=0.2)  # Z-axis (White)
    except ValueError:
        import pdb; pdb.set_trace()

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

def draw_pog(img, pog, color=(0,0,255), size=10):
    # Draw point of gaze (POG)
    x, y = pog
    cv2.circle(img, (int(x), int(y)), size, color, -1)
    return img

####################################################################################################
# 3D Rendering
####################################################################################################

def render_pog_with_screen(
        frame: np.ndarray,
        result: GazeResult,
        output_path: pathlib.Path,
        screen_RT: np.ndarray,
        screen_width_cm: float,
        screen_height_cm: float,
        screen_width_px: int,
        screen_height_px: int,
        gt_pog_px: Optional[np.ndarray] = None
    ):
    render_pog(frame, result, output_path, screen_RT, screen_width_cm, screen_height_cm)
    render_img = cv2.imread(str(output_path))
    screen_img = np.zeros([screen_height_px, screen_width_px, 3], dtype=np.float32)
    cv2.circle(screen_img, tuple(result.pog.pog_px.astype(np.int32)), 10, (0, 0, 255), -1)
    cv2.circle(screen_img, tuple(gt_pog_px.astype(np.int32)), 10, (0, 255, 0), -1)

    # Make the screen img match the same height as the render
    render_height = render_img.shape[0]
    new_width = int(screen_width_px * render_height / screen_height_px)
    screen_img = cv2.resize(screen_img, (new_width, render_height), interpolation=cv2.INTER_CUBIC)

    # Concatenate the images
    total_img = np.hstack((render_img, screen_img))
    return total_img