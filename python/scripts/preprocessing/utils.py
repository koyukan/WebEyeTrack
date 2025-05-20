import pathlib
import json

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from skimage.transform import PiecewiseAffineTransform, warp

from webeyetrack.vis import draw_axis, draw_landmarks_simple
from webeyetrack.utilities import vector_to_pitch_yaw, rotation_matrix_to_euler_angles, pitch_yaw_roll_to_gaze_vector


CWD = pathlib.Path(__file__).parent

IMG_SIZE = 512

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cv2tColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

# def vector_to_pitchyaw(vectors):
#     """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
#     n = vectors.shape[0]
#     out = np.empty((n, 2))
#     vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
#     out[:, 0] = np.arcsin(vectors[:, 1])  # theta
#     out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
#     return out

def data_normalization_entry(i, sample):

    # Select the image
    frame = sample['image']
    h, w, _ = frame.shape
    facial_landmarks = sample['facial_landmarks_2d']
    # facial_landmarks = sample['facial_landmarks'][:, :2] * np.array([w, h])
    detection_results = sample['facial_detection_results']
    draw_frame = frame.copy()

    # Draw the landmarks
    draw_landmarks_simple(draw_frame, facial_landmarks)
    # draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

    # Compute the homography matrix (4 pts) from the points to a final flat rectangle
    lefttop = facial_landmarks[103]
    leftbottom = facial_landmarks[150]
    righttop = facial_landmarks[332]
    rightbottom = facial_landmarks[379]
    center = facial_landmarks[4]

    src_pts = np.array([
        lefttop,
        leftbottom,
        rightbottom,
        righttop
    ], dtype=np.float32)

    # Add padding to the points, radially away from the center
    src_direction = src_pts - center
    src_pts = src_pts + np.array([0.4, 0.2]) * src_direction

    # if draw_frame is not None:
    #     for src_pt, color in zip(src_pts, [(0,0,0), (100, 100, 100), (200, 200, 200), (255, 255, 255)]):
    #         cv2.circle(draw_frame, tuple(src_pt.astype(np.int32)), 5, color, -1)

    dst_pts = np.array([
        [0, 0],
        [0, IMG_SIZE],
        [IMG_SIZE, IMG_SIZE],
        [IMG_SIZE, 0],
    ], dtype=np.float32)

    # Compute the homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts)
    warped_face_crop = cv2.warpPerspective(frame, M, (IMG_SIZE, IMG_SIZE))

    # Apply the homography matrix to the facial landmarks
    warped_facial_landmarks = np.dot(M, np.vstack((facial_landmarks.T, np.ones((1, facial_landmarks.shape[0])))))
    warped_facial_landmarks = (warped_facial_landmarks[:2, :] / warped_facial_landmarks[2, :]).T.astype(np.int32)

    # Generate crops for the eyes and each eye separately
    top_eyes_patch = warped_facial_landmarks[151]
    bottom_eyes_patch = warped_facial_landmarks[195]
    eyes_patch = warped_face_crop[top_eyes_patch[1]:bottom_eyes_patch[1], :]
    # left_eye_in_border = warped_facial_landmarks[193]
    # right_eye_in_border = warped_facial_landmarks[417]
    # left_eye_patch = eyes_patch[:, :left_eye_in_border[0]]
    # right_eye_patch = eyes_patch[:, right_eye_in_border[0]:]

    # Reshape the eyes patch to the same size (128, 512, 3)
    eyes_patch = cv2.resize(eyes_patch, (512, 128))

    g = sample['face_gaze_vector']
    n_g = vector_to_pitch_yaw(g)
    g_pitch, g_yaw, g_roll = n_g[0], n_g[1], 0
    rt = sample['facial_rt']
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rt[:3, :3])
    h_pitch, h_yaw, h_roll = -yaw, pitch, roll

    # Create the gaze and head pose vectors
    f_g = pitch_yaw_roll_to_gaze_vector(g_pitch, g_yaw, g_roll)
    f_h = pitch_yaw_roll_to_gaze_vector(h_pitch, h_yaw, h_roll)

    oh, ow = eyes_patch.shape[:2]

    # Basic visualization for debugging purposes
    if i % 25 == 0:
        # to_visualize = cv2.equalizeHist(cv2.cv2tColor(patch, cv2.COLOR_RGB2GRAY))
        to_visualize = cv2.cvtColor(eyes_patch.copy(), cv2.COLOR_RGB2BGR)
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_g,
        #                             length=80.0, thickness=1)
        to_visualize = draw_axis(to_visualize, g_pitch, g_yaw, 0, tdx=0.5 * ow, tdy=0.4 * oh, size=100, show_xy=True)
        to_visualize = draw_axis(to_visualize, h_pitch, h_yaw, 0, tdx=0.5 * ow, tdy=0.6 * oh, size=100, color=(0, 255, 0), show_xy=True)
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
        #                             length=40.0, thickness=3, color=(0, 0, 0))
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
        #                             length=40.0, thickness=1,
        #                             color=(255, 255, 255))
        cv2.imshow('frame', frame)
        cv2.imshow('warped_face_crop', warped_face_crop)
        cv2.imshow('draw_frame', draw_frame)
        cv2.imshow('normalized_patch', to_visualize)
        cv2.waitKey(0)

    return {
        'pixels': eyes_patch,
        'gaze_vector': f_g,
        'head_vector': f_h,

        # Take all of sample data
        **sample,
    }

def resize_intrinsics(intrinsic_matrix, original_size, new_size):
    """
    Adjusts the intrinsic matrix for a resized image.
    
    Parameters:
    - intrinsic_matrix: The original intrinsic matrix.
    - original_size: The original image size (width, height).
    - new_size: The new image size (width, height).
    
    Returns:
    - new_intrinsic_matrix: The adjusted intrinsic matrix for the resized image.
    """
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    new_intrinsic_matrix = intrinsic_matrix.copy()
    new_intrinsic_matrix[0, 0] *= scale_x  # fx
    new_intrinsic_matrix[1, 1] *= scale_y  # fy
    new_intrinsic_matrix[0, 2] *= scale_x  # cx
    new_intrinsic_matrix[1, 2] *= scale_y  # cy

    return new_intrinsic_matrix

def resize_annotations(annotations, original_size, new_size):

    # Any 2D point in the image plane needs to be scaled by the same factor
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    annotations.facial_landmarks_2d[0, :] *= scale_x
    annotations.facial_landmarks_2d[1, :] *= scale_y
    annotations.face_origin_2d[0] *= scale_x
    annotations.face_origin_2d[1] *= scale_y
    annotations.gaze_target_2d[0] *= scale_x
    annotations.gaze_target_2d[1] *= scale_y
    
    return annotations