import cv2
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import math
import trimesh
import imutils

from .data_protocols import GazeResult, EyeResult
from .model_based import vector_to_pitch_yaw, rotation_matrix_to_euler_angles
from .constants import *
from .utilities import (
    estimate_camera_intrinsics, 
    load_canonical_mesh, 
    load_3d_axis, 
    load_eyeball_model,
    load_camera_frustrum,
    load_screen_rect,
    load_pog_balls,
    load_gaze_vectors,
    transform_for_3d_scene,
    get_rotation_matrix_from_vector,
    euler_angles_to_rotation_matrix,
    transform_3d_to_3d,
    create_transformation_matrix,
    transform_3d_to_2d,
    OPEN3D_RT,
    OPEN3D_RT_SCREEN,
)

EYE_IMAGE_WIDTH = 400

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

def model_based_gaze_render(frame: np.ndarray, result: GazeResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]
    draw_frame = frame.copy()

    # Estimate the eyeball centers
    face_rt_copy = result.face_rt.copy()
    face_rt_copy[:3, 3] *= np.array([-1, -1, -1])

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

    # Compute the eyeball locations
    # Draw the eyeballs on the frame itsef
    named_sphere_points = {}
    for i, eye_result in enumerate([result.left, result.right]):
        eyeball_center = result.eyeball_centers[0] if i == 0 else result.eyeball_centers[1]
        name = 'left' if i == 0 else 'right'
        
        # Draw the eyeball as a 3D sphere
        # First, construct the 3D sphere within the canonical coordinate system
        sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=result.eyeball_radius)
        sphere_points = np.array(sphere_mesh.vertices)
        sphere_points += eyeball_center
        sphere_points_homogenous = np.hstack([sphere_points, np.ones((sphere_points.shape[0], 1))])
        sphere_points_transformed = (face_rt_copy @ sphere_points_homogenous.T).T

        # Project the 3D sphere to the 2D image
        height, width = frame.shape[:2]
        screen_sphere_homogenous = (result.perspective_matrix @ sphere_points_transformed.T).T
        screen_sphere_x_2d_n = screen_sphere_homogenous[:, 0] / screen_sphere_homogenous[:, 2]
        screen_sphere_y_2d_n = screen_sphere_homogenous[:, 1] / screen_sphere_homogenous[:, 2]
        screen_sphere_x_2d = (screen_sphere_x_2d_n + 1) * width / 2
        screen_sphere_y_2d = (1 - screen_sphere_y_2d_n) * height / 2

        screen_sphere_points = np.hstack([screen_sphere_x_2d.reshape(-1, 1), screen_sphere_y_2d.reshape(-1, 1)])
        named_sphere_points[name] = screen_sphere_points

        # Draw the 2D sphere
        for j in range(sphere_points.shape[0]):
            # cv2.circle(frame, (int(screen_sphere_x_2d[i]), int(screen_sphere_y_2d[i])), 1, (0, 0, 255), -1)
            prior_value = draw_frame[int(screen_sphere_y_2d[j]), int(screen_sphere_x_2d[j])]
            # Add a bit of grey to the color
            new_value = (prior_value + np.array([150, 150, 150])) / 2
            draw_frame[int(screen_sphere_y_2d[j]), int(screen_sphere_x_2d[j])] = new_value

        # Draw the eyeball center
        if i == 0:
            eyeball_center_2d = result.left.meta_data['eyeball_center_2d']
        else:
            eyeball_center_2d = result.right.meta_data['eyeball_center_2d']
        cv2.circle(draw_frame, tuple(eyeball_center_2d.astype(int)), 3, (0, 0, 255), -1)

        # Draw the iris center
        iris_center = left_2d_iris_px if i == 0 else right_2d_iris_px
        iris_center = iris_center[0]
        cv2.circle(draw_frame, tuple(iris_center.astype(int)), 3, (0, 255, 0), -1)

    eye_images = {}
    original_sizes = {}
    centroids = {}
    for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO
        centroids[i] = centroid

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        eye_result = result.left if i == 'left' else result.right
        eyeball_center = result.eyeball_centers[0] if i == 'left' else result.eyeball_centers[1]
        is_closed = eye_result.is_closed

        if width == 0 or height == 0:
            continue

        # Crop the eye
        eye_image = frame[
            int(centroid[1] - height/2):int(centroid[1] + height/2),
            int(centroid[0] - width/2):int(centroid[0] + width/2)
        ]

        eye_image_shape = eye_image.shape[:2]
        if eye_image_shape[0] == 0 or eye_image_shape[1] == 0:
            continue

        # Create eye image
        original_height, original_width = eye_image.shape[:2]
        original_sizes[i] = (original_width, original_height)

        # new_width, new_height = EYE_IMAGE_WIDTH, int(EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO)
        new_width, new_height = EYE_IMAGE_WIDTH, (EYE_IMAGE_WIDTH*original_height) // original_width
        eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        eye_images[i] = eye_image

        # Compute the shift and the scale
        shift_value = np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        scale_value = np.array([new_width/original_width, new_height/original_height])

        # Draw the outline of the eyelid
        shifted_eyelid_px = eyelid_total - shift_value
        prior_px = None
        for px in shifted_eyelid_px:
            resized_px = px * scale_value
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
            prior_px = resized_px
        
        # Draw the last line to close the loop
        resized_first_px = shifted_eyelid_px[0] * scale_value
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

        # Draw the eyeball
        screen_eye_points = named_sphere_points[i]
        for j in range(screen_eye_points.shape[0]):
            shifted_eyeball_pt = screen_eye_points[j] - shift_value
            resized_eyeball_pt = shifted_eyeball_pt * scale_value
            cv2.circle(eye_image, tuple(resized_eyeball_pt.astype(int)), 1, (200, 200, 200), -1)

        if is_closed:
            cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Draw the eyeball center
        eyeball_center_2d = eye_result.meta_data['eyeball_center_2d']
        shifted_eyeball_2d = eyeball_center_2d - shift_value
        scaled_eyeball_2d = shifted_eyeball_2d * scale_value
        cv2.circle(eye_image, scaled_eyeball_2d.astype(int), 5, (0, 255, 0), -1)
        
        # Convert 3D to pitch and yaw
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        draw_frame = draw_axis(draw_frame, -pitch, -yaw, 0, int(eyeball_center_2d[0]), int(eyeball_center_2d[1]), 100)
        eye_image = draw_axis(eye_image, -pitch, -yaw, 0, int(scaled_eyeball_2d[0]), int(scaled_eyeball_2d[1]), 100)

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - shift_value
        scaled_shifted_iris_px = shifted_iris_px * scale_value
        cv2.circle(eye_image, tuple(scaled_shifted_iris_px[0].astype(int)), 3, (0, 0, 255), -1)

    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(draw_frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the headpose on the frame
    # headrot = result.face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = result.face_origin_2d
    # frame = draw_axis(frame, pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # # Concatenate the images
    # if 'right' not in eye_images:
    #     right_eye_image = np.zeros((EYE_IMAGE_WIDTH, EYE_IMAGE_WIDTH, 3), dtype=np.uint8)
    # else:
    #     right_eye_image = eye_images['right']

    # if 'left' not in eye_images:
    #     left_eye_image = np.zeros((EYE_IMAGE_WIDTH, EYE_IMAGE_WIDTH, 3), dtype=np.uint8)
    # else:
    #     left_eye_image = eye_images['left']

    # Combine the eye images
    eye_combined = pad_and_concat_images(eye_images['right'], eye_images['left'])

    # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
    eyes_combined_resized = imutils.resize(eye_combined, width=frame.shape[1])
    
    # Concatenate the combined eyes image vertically with the frame
    total_frame = cv2.vconcat([draw_frame, eyes_combined_resized])

    # Pad the total frame with black at the bottom to avoid gittering when displaying
    total_height = total_frame.shape[0]
    padding_height = max(0, 800 - total_height)
    total_frame = cv2.copyMakeBorder(total_frame, 0, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return total_frame

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

def draw_axis(img, pitch, yaw, roll=0, tdx=None, tdy=None, size=100):
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
    # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 0), 3, tipLength=0.2)  # X-axis (Black)
    # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (100, 100, 100), 3, tipLength=0.2)  # Y-axis (Gray)
    try:
        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 255, 255), 3, tipLength=0.2)  # Z-axis (White)
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
    ):
    render_pog(frame, result, output_path, screen_RT, screen_width_cm, screen_height_cm)
    render_img = cv2.imread(str(output_path))
    screen_img = cv2.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)
    cv2.circle(screen_img, tuple(result.pog.pog_px.astype(np.int32)), 5, (0, 0, 255), -1)

    # Make the screen img match the same height as the render
    render_height = render_img.shape[0]
    ratio_h, ratio_w = render_height / screen_height_px, render_height / screen_width_px
    screen_height, screen_width = int(screen_height_px * ratio_h), int(screen_width_px * ratio_w)
    screen_img = cv2.resize(screen_img, (screen_width, screen_height))

    # Concatenate the images
    cv2.imwrite(str(output_path), np.hstack((render_img, screen_img)))
    # total_img = np.hstack((render_img, screen_img))
    # return total_img

def render_pog(
        frame: np.ndarray, 
        result: GazeResult, 
        output_path: pathlib.Path,
        screen_RT: np.ndarray,
        screen_width_cm: float,
        screen_height_cm: float,
    ):

    # Get the frame size 
    height, width = frame.shape[:2]
    max_size = max(height, width)
    w_ratio, h_ratio = width/max_size, height/max_size
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=width, height=height)
    visual.get_render_option().background_color = [1, 1, 1]
    visual.get_render_option().mesh_show_back_face = True

    # Change the z far to 1000
    vis = visual.get_view_control()
    vis.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    vis.convert_from_pinhole_camera_parameters(parameter=params)
    
    # Define intrinsics based on the frame
    intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)
    load_screen_rect(visual, screen_width_cm, screen_height_cm, screen_rt=screen_RT, scene_rt=OPEN3D_RT_SCREEN)
    load_camera_frustrum(w_ratio, h_ratio, visual, rt=OPEN3D_RT_SCREEN)
    left_pog, right_pog = load_pog_balls(visual)
    left_gaze_vector, right_gaze_vector = load_gaze_vectors(visual)
    
    camera_coordinate_axes = load_3d_axis(visual)
    points = [
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis end
        [0, 1, 0],  # Y-axis end
        [0, 0, 1],  # Z-axis end
    ]
    camera_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(np.array(points) * 5, OPEN3D_RT_SCREEN))
    visual.update_geometry(camera_coordinate_axes)

    # Compute face mesh
    face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face, OPEN3D_RT_SCREEN))
    new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
    face_mesh_lines.points = new_face_mesh_lines.points
    visual.update_geometry(face_mesh)
    visual.update_geometry(face_mesh_lines)

    # Update the 3d axes in the visualizer as well
    # Draw the canonical face axes by using the final_transform
    canonical_face_axes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) * 5
    camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
    face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d, OPEN3D_RT_SCREEN))
    visual.update_geometry(face_coordinate_axes)

    # Draw the 3D eyeball and gaze vector
    for i in ['left', 'right']:
        eye_result = result.left if i == 'left' else result.right
        origin = eye_result.origin
        direction = eye_result.direction
        pog_ball = left_pog if i == 'left' else right_pog
        gaze_vector = left_gaze_vector if i == 'left' else right_gaze_vector

        # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
        final_position = transform_for_3d_scene(origin.reshape((-1,3)), OPEN3D_RT_SCREEN).flatten()
        eyeball_meshes[i].translate(final_position, relative=False)

        # Rotation
        current_eye_R = eyeball_R[i]
        eye_R = get_rotation_matrix_from_vector(direction)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
        pitch, yaw, roll = yaw, pitch, roll # Flip the pitch and yaw
        eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)

        # Apply the scene transformation to the new eye rotation
        eye_R = np.dot(eye_R, OPEN3D_RT[:3, :3])

        # Compute the rotation matrix to rotate the current to the target
        new_eye_R = np.dot(eye_R, current_eye_R.T)
        eyeball_R[i] = eye_R
        eyeball_meshes[i].rotate(new_eye_R)
        visual.update_geometry(eyeball_meshes[i])

        # Draw the gaze vectors
        pts = np.array([origin, origin + direction * 100])
        transform_pts = transform_for_3d_scene(pts, OPEN3D_RT_SCREEN)
        gaze_vector.points = o3d.utility.Vector3dVector(transform_pts)
        gaze_vector.lines = o3d.utility.Vector2iVector([[0, 1]])
        if i == 'left':
            gaze_vector.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        else:
            gaze_vector.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
        visual.update_geometry(gaze_vector)

        # Position the PoG balls
        pog_position = transform_for_3d_scene(eye_result.pog.pog_cm_c.reshape((-1,3)), OPEN3D_RT_SCREEN).flatten()
        pog_ball.translate(pog_position, relative=False)
        visual.update_geometry(pog_ball)

    # Update visualizer
    visual.poll_events()
    visual.update_renderer()

    # Save the image
    visual.capture_screen_image(str(output_path))

    # Cleanup
    visual.destroy_window()
    print(f"3D render saved to {output_path}")

def render_3d_gaze_with_frame(frame: np.ndarray, result: GazeResult, output_path: pathlib.Path):
    render_3d_gaze(frame, result, output_path)
    render_img = cv2.imread(str(output_path))
    combined_frame = np.hstack([frame, render_img])
    return combined_frame

def render_3d_gaze(frame: np.ndarray, result: GazeResult, output_path: pathlib.Path) -> np.ndarray:

    # Get the frame size 
    height, width = frame.shape[:2]
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))
    
    # Initialize Open3D visual
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=width, height=height)
    visual.get_render_option().background_color = [0.1, 0.1, 0.1]
    visual.get_render_option().mesh_show_back_face = True
    visual.get_render_option().point_size = 10

    # Change the z far to 1000
    control = visual.get_view_control()
    control.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    control.convert_from_pinhole_camera_parameters(parameter=params)

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)

    # Compute face mesh
    face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face))
    new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
    face_mesh_lines.points = new_face_mesh_lines.points
    visual.update_geometry(face_mesh)
    visual.update_geometry(face_mesh_lines)

    # Draw canonical face axes
    canonical_face_axes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) * 5
    camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
    face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d))
    visual.update_geometry(face_coordinate_axes)

    # Compute the 3D eye origin
    for i in ['left', 'right']:
        eye_result = result.left if i == 'left' else result.right
        origin = eye_result.origin
        direction = eye_result.direction

        final_position = transform_for_3d_scene(origin.reshape((-1, 3))).flatten()
        eyeball_meshes[i].translate(final_position, relative=False)

        # Rotation
        current_eye_R = eyeball_R[i]
        eye_R = get_rotation_matrix_from_vector(direction)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
        pitch, yaw, roll = yaw, pitch, roll  # Flip pitch and yaw
        # pitch, yaw, roll = -yaw, pitch, roll  # Flip pitch and yaw
        eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)
        eye_R = np.dot(eye_R, OPEN3D_RT[:3, :3])

        # Compute the rotation matrix to rotate the current to the target
        new_eye_R = np.dot(eye_R, current_eye_R.T)
        eyeball_R[i] = eye_R
        eyeball_meshes[i].rotate(new_eye_R)
        visual.update_geometry(eyeball_meshes[i])

    # Render the scene
    visual.poll_events()
    visual.update_renderer()

    # Save the image
    visual.capture_screen_image(str(output_path))

    # Cleanup
    visual.close()
    visual.destroy_window()
    print(f"3D render saved to {output_path}")