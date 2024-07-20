import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

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