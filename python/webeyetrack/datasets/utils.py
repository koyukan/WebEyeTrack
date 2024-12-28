import pathlib
import json

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from skimage.transform import PiecewiseAffineTransform, warp

CWD = pathlib.Path(__file__).parent

def draw_landmarks_on_image(rgb_image, detection_result):
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

def compute_uv_texture(keypoints, img, texture_size=(244,244)):

  # uv_path = "./data/uv_map.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
  uv_path = CWD / 'data' / "uv_map.json"
  uv_map_dict = json.load(open(str(uv_path)))
  uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])

  #https://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html
  keypoints_uv = np.array([(texture_size[1]*x, texture_size[0]*y) for x,y in uv_map])

  tform = PiecewiseAffineTransform()
  tform.estimate(keypoints_uv,keypoints)
  texture = warp(img, tform, output_shape=(texture_size[1],texture_size[0]))
  texture = (255*texture).astype(np.uint8)

  return texture

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

def screen_plane_intersection(o, d, screen_R, screen_t):
    """
    Calculate the intersection of a gaze direction with a screen plane.
    
    Parameters:
    - o: Gaze origin (3D coordinates)
    - d: Gaze direction (3D coordinates)
    - screen_R: Rotation vector (Rodrigues vector) for the screen
    - screen_t: Translation vector for the screen
    
    Returns:
    - pog_mm: 2D point of gaze on the screen in millimeters (x, y)
    """

    # Obtain rotation matrix from the Rodrigues vector
    R_matrix, _ = cv2.Rodrigues(screen_R)  # screen_R should be a 3D vector (Rodrigues rotation)
    inv_R_matrix = np.linalg.inv(R_matrix)  # Inverse of the rotation matrix

    # Transform gaze origin and direction to screen coordinates
    o_s = np.dot(inv_R_matrix, (o - screen_t.T[0]))
    d_s = np.dot(inv_R_matrix, d)

    # Screen plane: z = 0 (assumed to be at origin with a normal vector along z-axis)
    a_s = np.array([0, 0, 0], dtype=np.float32)  # Point on the screen plane
    n_s = np.array([0, 0, 1], dtype=np.float32)  # Normal vector of the screen plane

    # Calculate the distance (lambda) to the screen plane
    lambda_ = np.dot(a_s - o_s, n_s) / np.dot(d_s, n_s)

    # Calculate the intersection point (3D)
    p = o_s + lambda_ * d_s

    # Keep only the x and y coordinates (2D point of gaze on screen)
    pog_mm = p[:2]

    return pog_mm

def screen_plane_intersection_2(o, d):
    """
    Calculate the intersection of a gaze direction with a screen plane.
    
    Parameters:
    - o: Gaze origin (3D coordinates)
    - d: Gaze direction (3D coordinates)
    
    Returns:
    - pog_mm: 2D point of gaze on the screen in millimeters (x, y)
    """

    # Screen plane: z = 0 (assumed to be at origin with a normal vector along z-axis)
    a = np.array([0, 0, 0], dtype=np.float32)  # Point on the screen plane
    n = np.array([0, 0, 1], dtype=np.float32)  # Normal vector of the screen plane

    # Calculate the distance (lambda) to the screen plane
    lambda_ = np.dot(a - o, n) / np.dot(d, n)

    # Calculate the intersection point (3D)
    p = o + lambda_ * d

    # Keep only the x and y coordinates (2D point of gaze on screen)
    pog_mm = p[:2]

    return pog_mm