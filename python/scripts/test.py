import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imutils
import math

from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis, core

CWD = pathlib.Path(__file__).parent
PYTHON_DIR = CWD.parent

LEFT_EYEAREA_LANDMARKS = [463, 359, 257, 253]

def reproject_2d_to_3d(u, v, z, intrinsics):
    """
    Reproject a 2D point (u, v) with a given depth z into 3D space.

    :param u: 2D x-coordinate (pixel)
    :param v: 2D y-coordinate (pixel)
    :param z: Depth value at (u, v)
    :param intrinsics: Camera intrinsic matrix (3x3)
    :return: 3D point in camera coordinates
    """
    # Create the 2D point in homogeneous coordinates
    uv_homogeneous = np.array([u, v, 1.0])

    # Invert the intrinsic matrix to map from image space to normalized camera coordinates
    inv_intrinsics = np.linalg.inv(intrinsics)

    # Reproject to 3D normalized coordinates
    normalized_coords = np.dot(inv_intrinsics, uv_homogeneous)

    # Scale by the depth (z) to get the 3D coordinates in camera space
    X = normalized_coords * z

    return X

if __name__ == '__main__':
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)

    # Setup MediaPipe Face Landmark model
    base_options = python.BaseOptions(model_asset_path=str(PYTHON_DIR / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Load the frames and draw the landmarks
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define intrinsics based on the frame dimensions
        height, width = frame.shape[:2]
        focal_length = width  # Assuming fx = fy = width for simplicity, adjust based on real camera
        intrinsics = np.array([[focal_length, 0, width // 2], 
                               [0, focal_length, height // 2], 
                               [0, 0, 1]])

        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)

        # Ensure there is at least one face detected
        if len(detection_results.face_landmarks) == 0:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Draw the canonical face origin xyz=[0,0,0] via the facial transformation matrix
        face_rt = detection_results.facial_transformation_matrixes[0]
     
        # Draw the landmarks
        frame = draw_landmarks_on_image(frame, detection_results)

        # use the nose as the face origin
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in detection_results.face_landmarks[0]])
        # x, y, z = detection_results.face_landmarks[0][1].x * width, detection_results.face_landmarks[0][1].y * height, detection_results.face_landmarks[0][1].z
        x, y, z = face_landmarks[1, :3] * np.array([width, height, 1])

        # Draw the rotation matrix
        rotation_matrix = face_rt[:3, :3].copy()

        # Convert to pitch, yaw, roll
        pitch, yaw, roll = vis.rotation_matrix_to_euler_angles(rotation_matrix)
        pitch, yaw, roll = yaw, pitch, -roll
        # pitch, yaw, roll = yaw, pitch, roll
        # pitch, yaw, roll = 0, 0, 0
        frame = vis.draw_axis(frame, pitch, yaw, roll, int(x), int(y), 100)
        # frame = vis.draw_axis_from_rotation_matrix(frame, rotation_matrix, x, y)

        # Create a new rotatiom matrix
        new_rotation_matrix = core.euler_angles_to_rotation_matrix(pitch, yaw, roll)
        new_face_rt = np.eye(4)
        new_face_rt[:3, :3] = new_rotation_matrix
        # translation = face_rt[:3, 3]
        # translation *= np.array([-1, 1, 1])
        # translation = np.array([0, 0, -5])

        # Compute the translation via reprojecting the nose as the origin
        # translation = reproject_2d_to_3d(x, y, face_rt[2, 3], intrinsics)
        translation = reproject_2d_to_3d(x, y, z, intrinsics)

        # Compute the average left eye area
        left_eye_landmarks = face_landmarks[LEFT_EYEAREA_LANDMARKS]
        left_eye_landmarks = left_eye_landmarks[:, :3]
        left_eye_center = np.mean(left_eye_landmarks, axis=0) / 75

        new_face_rt[:3, 3] = translation
        # face_origin = np.array([2.2, 2.5, 3, 1])  # 3D origin point in canonical face space
        # face_origin = np.array([1e-3, 0, 0, 1])  # 3D origin point in canonical face space
        # print(left_eye_center)
        face_origin = np.array([left_eye_center[0], left_eye_center[1], left_eye_center[2], 1])  # 3D origin point in canonical face space
        
        # Transform the face origin from canonical to world space
        face_origin_3d = np.dot(new_face_rt, face_origin)
        face_origin_3d = face_origin_3d[:3] / face_origin_3d[3]  # Homogeneous to Cartesian coordinates

        # Project the point from 3D world coordinates to 2D image plane
        face_origin = np.dot(intrinsics, face_origin_3d)
        face_origin = face_origin[:2] / face_origin[2]  # Perspective divide to get 2D coordinates

        # Draw the projected point on the frame
        cv2.circle(frame, (int(face_origin[0]), int(face_origin[1])), 5, (0, 0, 255), -1)
        # print(face_origin_3d[2], detection_results.face_landmarks[0][1].z)


        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
