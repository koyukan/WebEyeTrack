import time
from typing import Dict, Any, Literal, Optional, Tuple
import math

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skopt import gp_minimize
from skopt.space import Real

from webeyetrack.datasets.utils import screen_plane_intersection, screen_plane_intersection_2
from ..vis import draw_axis
from ..core import pitch_yaw_to_gaze_vector, rotation_matrix_to_euler_angles, vector_to_pitch_yaw
from ..data_protocols import FLGEResult, EyeResult

LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]
LEFT_BLENDSHAPES = [14, 16, 18, 12]
RIGHT_BLENDSHAPES = [13, 15, 17, 11]
REAL_WORLD_IPD_CM = 6.3 # Inter-pupilary distance (cm)
HFOV = 100
VFOV = 90

# Landmark reference:
# https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png

# Format [leftmost, rightmost, topmost, bottommost]
LEFT_EYEAREA_LANDMARKS = [463, 359, 257, 253]
RIGHT_EYEAREA_LANDMARKS = [130, 243, 27, 23]
LEFT_EYEAREA_TOTAL_LANDMARKS = [463,  341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414]
RIGHT_EYEAREA_TOTAL_LANDMARKS = [130, 25,  110, 24,  23,  22,  26,  112, 243, 190, 56,  28,  27,  29,  30,  247]

# Format [leftmost, rightmost, topmost, bottommost]
LEFT_EYELID_LANDMARKS = [362, 263, 386, 374]
RIGHT_EYELID_LANDMARKS = [33, 133, 159, 145]
LEFT_EYELID_TOTAL_LANDMARKS = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYELID_TOTAL_LANDMARKS = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# EAR landmarks (for detecting eye blinking) # p1, p2, p3, p4, p5, p6
LEFT_EYE_EAR_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_LANDMARKS = [133, 158, 160, 33, 144, 153]

RIGHT_IRIS_LANDMARKS = [468, 470, 469, 472, 471] # center, top, right, botton, left
LEFT_IRIS_LANDMARKS = [473, 475, 474, 477, 476] # center, top, right, botton, left

# EYE_PADDING_HEIGHT = 0.1
EYE_PADDING_WIDTH = 0.3
EYE_HEIGHT_RATIO = 0.7

# Position of eyeball center based on canonical coordinate system
# LEFT_EYEBALL_CENTER = np.array([3.0278, -2.7526, 2.7234]) * 10 # X, Y, Z
# RIGHT_EYEBALL_CENTER = np.array([-3.0278, -2.7526, 2.7234]) * 10 # X, Y, Z

# Average radius of an eyeball in cm
EYEBALL_RADIUS = 1
EYEBALL_X, EYEBALL_Y, EYEBALL_Z = 3, 2.7, 3
EYEBALL_DEFAULT = (np.array([-EYEBALL_X, -EYEBALL_Y, -EYEBALL_Z]), np.array([EYEBALL_X, -EYEBALL_Y, -EYEBALL_Z])) # left, right

# According to https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/graphs/face_effect/face_effect_gpu.pbtxt#L61-L65
VERTICAL_FOV_DEGREES = 60
NEAR = 1.0 # 1cm
FAR = 10000 # 100m 
ORIGIN_POINT_LOCATION = 'BOTTOM_LEFT_CORNER'

def compute_2d_origin(points):
    (cx, cy), radius = cv2.minEnclosingCircle(points.astype(np.float32))
    center = np.array([cx, cy], dtype=np.int32)
    return center

def create_perspective_matrix(aspect_ratio):
    k_degrees_to_radians = np.pi / 180.0

    # Initialize a 4x4 matrix filled with zeros
    perspective_matrix = np.zeros((4, 4), dtype=np.float32)

    # Standard perspective projection matrix calculations
    f = 1.0 / np.tan(k_degrees_to_radians * VERTICAL_FOV_DEGREES / 2.0)
    denom = 1.0 / (NEAR - FAR)

    # Populate the matrix values
    perspective_matrix[0, 0] = f / aspect_ratio
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = (NEAR + FAR) * denom
    perspective_matrix[2, 3] = -1.0
    perspective_matrix[3, 2] = 2.0 * FAR * NEAR * denom

    # Flip Y-axis if origin point location is top-left corner
    if ORIGIN_POINT_LOCATION == 'TOP_LEFT_CORNER':
        perspective_matrix[1, 1] *= -1.0

    return perspective_matrix

def convert_uv_to_xyz(perspective_matrix, u, v, z_relative):
    # Step 1: Convert normalized (u, v) to Normalized Device Coordinates (NDC)
    ndc_x = 2 * u - 1
    ndc_y = 1 - 2 * v

    # Step 2: Create the NDC point in homogeneous coordinates
    ndc_point = np.array([ndc_x, ndc_y, -1.0, 1.0])

    # Step 3: Invert the perspective matrix to go from NDC to world space
    inv_perspective_matrix = np.linalg.inv(perspective_matrix)

    # Step 4: Compute the point in world space (in homogeneous coordinates)
    world_point_homogeneous = np.dot(inv_perspective_matrix, ndc_point)

    # Step 5: Dehomogenize (convert from homogeneous to Cartesian coordinates)
    x = world_point_homogeneous[0] / world_point_homogeneous[3]
    y = world_point_homogeneous[1] / world_point_homogeneous[3]
    z = world_point_homogeneous[2] / world_point_homogeneous[3]

    # Step 6: Scale using the relative depth
    # Option A
    x_relative = -x #* z_relative
    y_relative = y #* z_relative
    # z_relative = z * z_relative

    # Option B
    # x_relative = x * z_relative
    # y_relative = y * z_relative
    # z_relative = z * z_relative

    return np.array([x_relative, y_relative, z_relative])

def convert_xyz_to_uv(perspective_matrix, x, y, z):
    # Step 1: Convert (x, y, z) to homogeneous coordinates (x, y, z, 1)
    world_point = np.array([x, -y, z, 1.0])
    # world_point = np.array([x, y, z, 1.0])

    # Step 2: Apply the perspective projection matrix
    ndc_point_homogeneous = np.dot(perspective_matrix, world_point)

    # Step 3: Dehomogenize to convert from homogeneous to Cartesian coordinates
    u_ndc = ndc_point_homogeneous[0] / ndc_point_homogeneous[3]
    v_ndc = ndc_point_homogeneous[1] / ndc_point_homogeneous[3]
    z_ndc = ndc_point_homogeneous[2] / ndc_point_homogeneous[3]

    # Step 4: Convert from NDC to normalized coordinates (u, v) in the range [0, 1]
    u = (u_ndc + 1) / 2
    v = (1 - v_ndc) / 2

    return u, v

def convert_xyz_to_uv_with_intrinsic(intrinsic_matrix, x, y, z):
    # Step 1: Create the 3D point in homogeneous coordinates
    point_3d = np.array([-x, -y, z, 1.0])

    # Step 2: Project the 3D point to the image plane using the intrinsic matrix
    # Remove the homogeneous component before applying K
    point_3d_camera = point_3d[:3]  # Only use x, y, z

    # Apply the intrinsic matrix to project to 2D
    projected_point_homogeneous = np.dot(intrinsic_matrix, point_3d_camera)

    # Step 3: Dehomogenize to convert to Cartesian coordinates (u, v)
    u = projected_point_homogeneous[0] / projected_point_homogeneous[2]
    v = projected_point_homogeneous[1] / projected_point_homogeneous[2]

    return np.array([u, v])

# Facial Landmark Gaze Estimation
class FLGE():

    def __init__(
            self, 
            model_asset_path: str, 
            gaze_direction_estimation: Literal['model-based', 'landmark2d', 'blendshape'] = 'model-based', 
            eyeball_centers: Tuple[np.ndarray, np.ndarray] = EYEBALL_DEFAULT,
            eyeball_radius: float = EYEBALL_RADIUS,
            ear_threshold: float = 0.1
        ):

        # Saving options
        self.gaze_direction_estimation = gaze_direction_estimation

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Create perspecive matrix variable
        self.perspective_matrix: Optional[np.ndarray] = None
        self.inv_perspective_matrix: Optional[np.ndarray] = None

        # Store default parameters
        self.eyeball_centers = eyeball_centers
        self.eyeball_radius = eyeball_radius
        self.ear_threshold = ear_threshold

        # Gaze filter
        self.prior_gaze = None
        self.prior_depth = None

    def calibrate(self, samples):

        def loss_fn():
            ...

        dimensions = [
            Real(15/2, 25/2, name='eyeball_radius'),
            Real(EYEBALL_X/2, EYEBALL_X*2, name='eyeball_x'),
            Real(EYEBALL_Y/2, EYEBALL_Y*2, name='eyeball_y'),
            Real(EYEBALL_Z/2, EYEBALL_Z*2, name='eyeball_z')
        ]

        # Initial guess for the parameters
        x0 = [EYEBALL_RADIUS, EYEBALL_X, EYEBALL_Y, EYEBALL_Z]

        # Perform the optimization
        

    def estimate_inter_pupillary_distance_2d(self, facial_landmarks, height, width):
        data_2d_pairs = {
            'left': facial_landmarks[LEFT_EYE_LANDMARKS][:, :2] * np.array([width, height]),
            'right': facial_landmarks[RIGHT_EYE_LANDMARKS][:, :2] * np.array([width, height])
        }
        data_3d_pairs = {
            'left': facial_landmarks[LEFT_EYE_LANDMARKS][:, :3],
            'right': facial_landmarks[RIGHT_EYE_LANDMARKS][:, :3]
        }

        # Compute the 2D eye origin
        origins_2d = {}
        for k,v in data_2d_pairs.items():
            origins_2d[k] = compute_2d_origin(v)

        # Compute the 3D eye origin
        origins_3d = {}
        for k, v in data_3d_pairs.items():
            origins_3d[k] = np.mean(v, axis=0)

        # Compute the scaling factor between mediapipe canonical & world coordinate
        l, r = origins_3d['left'], origins_3d['right']
        canonical_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1],2) + np.power(l[2] - r[2], 2))

        # Compute the distance in 2d 
        l, r = origins_2d['left'], origins_2d['right']
        image_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1], 2))

        positions = {
            'eye_origins_2d': {
                'left': l,
                'right': r
            },
            'eye_origins_3d_canonical': {
                'left': origins_3d['left'],
                'right': origins_3d['right']
            }
        }
        distances = {
            'canonical_ipd_3d': canonical_ipd,
            'image_ipd': image_ipd
        }

        return (positions, distances)

    def estimate_gaze_vector_based_on_model_based(self, frame, facial_landmarks, face_rt, height, width):
        
        # Estimate the eyeball centers
        face_rt_copy = face_rt.copy()
        face_rt_copy[:3, 3] *= np.array([-1, -1, -1])

        # Must for gaze estimation
        gaze_vectors = {}
        eye_closed = {}

        # Visualization for debug
        eyeball_center_2d = {'left': None, 'right': None}
        eyeball_radius_2d = {'left': None, 'right': None}

        for i, canonical_eyeball in zip(['left', 'right'], self.eyeball_centers):
            # if i == 'right':
            #     eye_closed[i] = True
            #     continue

            # Convert to homogenous
            eyeball_homogeneous = np.append(canonical_eyeball, 1)

            # Convert from canonical to camera space
            camera_eyeball = face_rt_copy @ eyeball_homogeneous
            sphere_center = camera_eyeball[:3]

            # Obtain the 2D eyeball center and radius
            screen_landmark_homogenous = self.perspective_matrix @ camera_eyeball
            eyeball_x_2d_n = screen_landmark_homogenous[0] / screen_landmark_homogenous[2]
            eyeball_y_2d_n = screen_landmark_homogenous[1] / screen_landmark_homogenous[2]
            eyeball_x_2d = (eyeball_x_2d_n + 1) * width / 2
            eyeball_y_2d = (eyeball_y_2d_n * -1 + 1) * height / 2
            eyeball_center_2d[i] = np.array([eyeball_x_2d, eyeball_y_2d])
            eyeball_radius_2d[i] = 0.85 * (500/camera_eyeball[2])

            # Draw the eyeball center and radius
            # cv2.circle(frame, (int(eyeball_x_2d), int(eyeball_y_2d)), 2, (0, 0, 255), -1)
            
            # First, determine if the eye is closed, by computing the EAR
            # EAR = ||p_2 - p_6|| + ||p_3 - p_5|| / (2 * ||p_1 - p_4||)
            EYE_EAR_LANDMARKS = LEFT_EYE_EAR_LANDMARKS if i == 'left' else RIGHT_EYE_EAR_LANDMARKS
            p1 = facial_landmarks[EYE_EAR_LANDMARKS[0], :2]
            p2 = facial_landmarks[EYE_EAR_LANDMARKS[1], :2]
            p3 = facial_landmarks[EYE_EAR_LANDMARKS[2], :2]
            p4 = facial_landmarks[EYE_EAR_LANDMARKS[3], :2]
            p5 = facial_landmarks[EYE_EAR_LANDMARKS[4], :2]
            p6 = facial_landmarks[EYE_EAR_LANDMARKS[5], :2]

            # Draw all the EAR landmarks
            # for j, landmark in enumerate(EYE_EAR_LANDMARKS):
            #     x, y = facial_landmarks[landmark, :2]
            #     cv2.circle(frame, (int(x * width), int(y * height)), 2, (0, 255, 0), -1)
            #     cv2.putText(frame, f"p{j+1}", (int(x * width), int(y * height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))
            eye_closed[i] = False
            if ear < self.ear_threshold:
                eye_closed[i] = True
                continue

            # Compute the 3D pupil by using a line-sphere intersection problem
            # Reference: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
            # Convert from 0-1 to -1 to 1
            pupil = facial_landmarks[LEFT_IRIS_LANDMARKS[0], :3] if i == 'left' else facial_landmarks[RIGHT_IRIS_LANDMARKS[0], :3]
            pupil2d = np.array([pupil[0] * width, pupil[1] * height])
            ndc_y = 1 - (2 * pupil2d[1] / height)
            ndc_x = (2 * pupil2d[0] / width) - 1
            ndc_point = np.array([ndc_x, ndc_y, -1.0, 1.9])
            
            # Draw the pupil
            # cv2.circle(frame, (int(pupil2d[0]), int(pupil2d[1])), 2, (0, 255, 0), -1)

            # Compute the ray in 3D space
            world_point_homogeneous = np.dot(self.inv_perspective_matrix, ndc_point)
            world_point = world_point_homogeneous[:3] / world_point_homogeneous[3]
            ray_direction = world_point - np.array([0, 0, 0])
            ray_direction /= np.linalg.norm(ray_direction)  # Normalize the direction
            
            # Camera origin and Calculate intersection with the sphere
            camera_origin = np.array([0.0, 0.0, 0.0])
            oc = camera_origin - sphere_center

            # Solve the quadratic equation ax^2 + bx + c = 0
            discriminant = np.dot(ray_direction, oc) ** 2 - (np.dot(oc, oc) - self.eyeball_radius ** 2)

            if discriminant < 0:
                # No real intersections
                # cv2.imshow('frame', imutils.resize(frame, width=1000))

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                continue
                # return None

            # Calculate the two possible intersection points
            t1 = np.dot(-ray_direction, oc) - np.sqrt(discriminant)
            t2 = np.dot(-ray_direction, oc) + np.sqrt(discriminant)

            # We are interested in the first intersection that is in front of the camera
            pupil_3d = None
            if t1 > t2:
                pupil_3d = camera_origin + t1 * ray_direction
            else:
                pupil_3d = camera_origin + t2 * ray_direction

            # Compute the gaze direction based on the eyeball center and 3D pupil
            gaze_vector = pupil_3d - sphere_center
            gaze_vector /= np.linalg.norm(gaze_vector)

            # DEBUG: For debugging purposes, make the gaze vector straight to the z-axis
            # gaze_vector = np.array([0, 0, -1])
            # gaze_vector = np.array([-0.1, 0.0, -0.9])
            # gaze_vector /= np.linalg.norm(gaze_vector)
            # gaze_vector = np.array([-0.04122334, -0.25422794, -0.96626538])

            # Convert gaze vector to pitch and yaw to correct
            pitch, yaw = vector_to_pitch_yaw(gaze_vector)
            pitch, yaw = -pitch, yaw
            gaze_vector = pitch_yaw_to_gaze_vector(pitch, yaw)

            # Store
            gaze_vectors[i] = gaze_vector

        # Compute the average gaze vector
        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
            face_gaze_vector /= np.linalg.norm(face_gaze_vector)
        elif 'left' in gaze_vectors:
            face_gaze_vector = gaze_vectors['left']
            gaze_vectors['right'] = np.array([0,0,-1])
        elif 'right' in gaze_vectors:
            face_gaze_vector = gaze_vectors['right']
            gaze_vectors['left'] = np.array([0,0,-1])
        else:
            face_gaze_vector = np.array([0,0,-1])
            gaze_vectors['left'] = np.array([0,0,-1])
            gaze_vectors['right'] = np.array([0,0,-1])

        # Debugging purposes
        # cv2.imshow('debug_frame', frame)

        return {
            'face': face_gaze_vector,
            'eyes': {
                'is_closed': eye_closed,
                'vector': gaze_vectors,
                'meta_data': {
                    'left': {
                        'eyeball_center_2d': eyeball_center_2d['left'],
                        'eyeball_radius_2d': eyeball_radius_2d['left']
                    },
                    'right': {
                        'eyeball_center_2d': eyeball_center_2d['right'],
                        'eyeball_radius_2d': eyeball_radius_2d['right']
                    }
                }
            }
        }

    def estimate_gaze_vector_based_on_eye_landmarks(self, frame, facial_landmarks, face_rt, height, width):

        # Compute the bbox by using the edges of the each eyes
        left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([height, width])
        left_2d_eyelid_px = facial_landmarks[LEFT_EYELID_LANDMARKS, :2] * np.array([height, width])
        left_2d_iris_px = facial_landmarks[LEFT_IRIS_LANDMARKS, :2] * np.array([height, width])
        
        right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([height, width])
        right_2d_eyelid_px = facial_landmarks[RIGHT_EYELID_LANDMARKS, :2] * np.array([height, width])
        right_2d_iris_px = facial_landmarks[RIGHT_IRIS_LANDMARKS, :2] * np.array([height, width])

        # Apply face_rt to the EYEBALL_CENTERs to get the 3D position
        # canonical_lefteye_center_homo = np.append(LEFT_EYEBALL_CENTER, 1)
        # canonical_righteye_center_homo = np.append(RIGHT_EYEBALL_CENTER, 1)
        # left_eye_ball_center = np.dot(face_rt[:3, :3], LEFT_EYEBALL_CENTER) + face_rt[:3, 3]
        # right_eye_ball_center = np.dot(face_rt[:3, :3], RIGHT_EYEBALL_CENTER) + face_rt[:3, 3]

        # tf_lefteye_center_homo = face_rt @ canonical_lefteye_center_homo
        # tf_lefteye_center = tf_lefteye_center_homo[:3] / tf_lefteye_center_homo[-1]
        # u_normalized = tf_lefteye_center[0] / width
        # v_normalized = tf_lefteye_center[1] / height
        # z_relative = tf_lefteye_center[2] / tf_lefteye_center[0]
        # actual_UVZ = facial_landmarks[LEFT_EYEAREA_LANDMARKS[0], :3]
        # UVZ = (u_normalized, v_normalized, z_relative)

        # 3D
        left_eye_fl = facial_landmarks[LEFT_EYELID_LANDMARKS, :3]
        right_eye_fl = facial_landmarks[RIGHT_EYELID_LANDMARKS, :3]

        left_landmarks = [
            left_2d_eye_px, 
            left_2d_eyelid_px,
        ]
        right_landmarks = [
            right_2d_eye_px, 
            right_2d_eyelid_px,
        ]

        eye_closed = {}
        eye_images = {}
        gaze_vectors = {}
        gaze_origins_2d = {}
        headpose_corrected_eye_center = {}
        for i, (eye, eyelid) in {'left': left_landmarks, 'right': right_landmarks}.items():
            centroid = np.mean(eye, axis=0)
            actual_width = np.abs(eye[1,0] - eye[0, 0])
            width = actual_width * (1 + EYE_PADDING_WIDTH)
            height = width * EYE_HEIGHT_RATIO

            gaze_origins_2d[i] = centroid

            # Determine if closed by the eyelid
            eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
            eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
            is_closed = False

            # Determine if the eye is closed by the ratio of the height based on the width
            if eyelid_height / eyelid_width < 0.05:
                is_closed = True

            if width == 0 or height == 0:
                continue

            # Draw if the eye is closed on the top left corner
            eye_closed[i] = is_closed
            if is_closed:
                continue

            # Shift the IRIS landmarks to the cropped eye
            iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
            shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
            iris_center = shifted_iris_px[0]
            eye_center = np.array([width/2, height/2])

            # Compute the radius of the iris
            left_iris_radius = np.linalg.norm(iris_center - shifted_iris_px[2])
            right_iris_radius = np.linalg.norm(iris_center - shifted_iris_px[4])
            iris_radius = np.mean([left_iris_radius, right_iris_radius]) # 10

            # Shift the eye center by the headpose
            headrot = face_rt[:3, :3]
            pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
            pitch, yaw = yaw, -pitch # Swap the pitch and yaw
            size = actual_width / 4
            pitch = (pitch * np.pi / 180)
            yaw = (yaw * np.pi / 180)
            x3 = size * (math.sin(yaw))
            y3 = size * (-math.cos(yaw) * math.sin(pitch))
            # frame = draw_axis(frame, -pitch, yaw, 0, int(face_origin[0]), int(face_origin[1]), 100)

            old_iris_center = iris_px[0]
            # cv2.circle(frame, (int(old_iris_center[0]), int(old_iris_center[1])), 2, (0, 0, 255), -1)
            shifted_iris_center = old_iris_center + np.array([int(x3), int(y3)])
            # cv2.circle(frame, (int(shifted_iris_center[0]), int(shifted_iris_center[1])), 2, (0, 255, 0), -1)
            # cv2.line(frame, (int(old_iris_center[0]), int(old_iris_center[1])), (int(shifted_iris_center[0]), int(shifted_iris_center[1])), (0, 255, 0), 1)

            # Shifting the eye_center by the headpose
            # print(f"Eye center: {eye_center}, shift: {np.array([x3, y3])}, new: {eye_center + np.array([x3, y3])}")
            eye_center = eye_center + np.array([x3, y3])
            headpose_corrected_eye_center[i] = eye_center

            # Based on the direction and magnitude of the line, compute the gaze direction
            # Compute 2D vector from eyeball center to iris center
            # gaze_vector_2d = shifted_iris_px[0] - iris_px[0]
            gaze_vector_2d = iris_center - eye_center
            # gaze_vector_2d = np.array([0,0])

            # Estimate the depth (Z) based on the 2D vector length
            # z_depth = EYEBALL_RADIUS / np.linalg.norm(gaze_vector_2d)
            z_depth = 2.0
            # Estimate the depth (Z) based on the size of the iris
            # z_depth = EYEBALL_RADIUS

            # Compute yaw (horizontal rotation)
            yaw = np.arctan2(gaze_vector_2d[0] / iris_radius, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Compute pitch (vertical rotation)
            pitch = np.arctan2(gaze_vector_2d[1] / iris_radius, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Convert the pitch and yaw to a 3D vector
            gaze_vector = pitch_yaw_to_gaze_vector(pitch, yaw)
            gaze_vectors[i] = gaze_vector

            # # Compute 3D gaze origin
            # eye_fl = left_eye_fl if i == 'left' else right_eye_fl
            # gaze_origin = np.mean(eye_fl, axis=0)
            # gaze_origins[i] = gaze_origin

        # Compute average gaze origin 2d
        face_origin_2d = (gaze_origins_2d['left'] + gaze_origins_2d['right']) / 2

        # Draw the headpose on the frame
        # headrot = face_rt[:3, :3]
        # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
        # pitch, yaw = yaw, pitch
        # face_origin = face_origin_2d
        # frame = draw_axis(frame, -pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()

        # Compute the average gaze vector
        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
            face_gaze_vector /= np.linalg.norm(face_gaze_vector)
        elif 'left' in gaze_vectors:
            face_gaze_vector = gaze_vectors['left']
            gaze_vectors['right'] = np.array([0,0,1])
            headpose_corrected_eye_center['right'] = None
        elif 'right' in gaze_vectors:
            face_gaze_vector = gaze_vectors['right']
            gaze_vectors['left'] = np.array([0,0,1])
            headpose_corrected_eye_center['left'] = None
        else:
            face_gaze_vector = np.array([0,0,1])
            gaze_vectors['left'] = np.array([0,0,1])
            gaze_vectors['right'] = np.array([0,0,1])
            headpose_corrected_eye_center['left'] = None
            headpose_corrected_eye_center['right'] = None

        return {
            'face': face_gaze_vector,
            'eyes': {
                'is_closed': eye_closed,
                'vector': gaze_vectors,
                'meta_data': {
                    'left': {
                        'headpose_corrected_eye_center': headpose_corrected_eye_center['left']
                    },
                    'right': {
                        'headpose_corrected_eye_center': headpose_corrected_eye_center['right']
                    }
                }
            }
        }

    def estimate_gaze_vector_based_on_eye_blendshapes(self, face_blendshapes, face_rt):

        # Get the transformation matrix
        # Invert the y and z axis
        transform = face_rt.copy()
        transform = np.diag([-1, 1, 1, 1]) @ transform
        
        # Compute the iris direction
        gaze_directions = {}
        for option, value in {'left': LEFT_BLENDSHAPES, 'right': RIGHT_BLENDSHAPES}.items():
            blendshapes = face_blendshapes
            look_in, look_out, look_up, look_down = ([blendshapes[i] for i in value])
            hfov = np.deg2rad(HFOV)
            vfov = np.deg2rad(VFOV)

            rx = hfov * 0.5 * (look_down - look_up)
            ry = vfov * 0.5 * (look_in - look_out) * (1 if option == 'left' else -1)

            # Create euler angle
            euler_angles = np.array([rx, ry, 0])

            # # Convert to rotation matrix
            rotation_matrix = cv2.Rodrigues(euler_angles)[0]

            # Compute the gaze direction
            gaze_directions[option] = rotation_matrix

        # Apply the rotation to the gaze direction
        for k, v in gaze_directions.items():
            rotation_matrix = transform[:3, :3]
            gaze_directions[k] = v.dot(rotation_matrix)

        # Compute the gaze direction by apply the rotation to a [0,0,-1] vector
        gaze_vectors = {
            'left': np.array([0,0,-1]),
            'right': np.array([0,0,-1])
        }
        for k, v in gaze_directions.items():
            gaze_vectors[k] = v.dot(gaze_vectors[k])

        # Compute the average gaze vector
        face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
        face_gaze_vector /= np.linalg.norm(face_gaze_vector)

        return {
            'face': face_gaze_vector,
            'eyes': {
                'is_closed': {'left': False, 'right': False},
                'vector': gaze_vectors,
                'meta_data': {
                    'left': {},
                    'right': {}
                }
            }            
        }

    def estimate_2d_3d_eye_face_origins(self, frame, facial_landmarks, face_rt, height, width, intrinsics):

        # First, compute the inter-pupillary distance
        positions, distances = self.estimate_inter_pupillary_distance_2d(
            facial_landmarks, 
            height, 
            width
        )

        # Estimate the scale
        metric_scale = REAL_WORLD_IPD_CM * 10 / distances['canonical_ipd_3d']

        # Convert uvz to xyz
        relative_face_mesh = np.array([convert_uv_to_xyz(self.perspective_matrix, x[0], x[1], x[2]) for x in facial_landmarks[:, :3]])
        centroid = relative_face_mesh.mean(axis=0)
        demeaned_relative_face_mesh = relative_face_mesh.copy() # - centroid
        
        data_3d_pairs = {
            'left': demeaned_relative_face_mesh[LEFT_EYE_LANDMARKS][:, :3],
            'right': demeaned_relative_face_mesh[RIGHT_EYE_LANDMARKS][:, :3]
        }

        # Compute the 3D eye origin
        origins_3d = {}
        for k, v in data_3d_pairs.items():
            origins_3d[k] = np.mean(v, axis=0)

        # Compute the scaling factor between mediapipe per-world & world mm coordinate
        l, r = origins_3d['left'], origins_3d['right']
        per_frame_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1],2) + np.power(l[2] - r[2], 2))
        scale = (10 * REAL_WORLD_IPD_CM) / per_frame_ipd

        # Compute the depth
        theta = np.arctan(face_rt[0, 2] / face_rt[2, 2])
        focal_length_pixels = 1 / np.tan(np.deg2rad(VERTICAL_FOV_DEGREES) / 2) * height / 2
        depth_mm = (focal_length_pixels * REAL_WORLD_IPD_CM * 10 * np.cos(theta)) / distances['image_ipd'] * 2.25
        if self.prior_depth is not None:
            depth_mm = (depth_mm + self.prior_depth) / 2
        self.prior_depth = depth_mm
        # print(f"Depth: {depth_mm}")

        # Apply the scale
        scaled_demeaned_relative_face_mesh = demeaned_relative_face_mesh * scale

        # Returned to the position
        translation = np.array([0, 0, depth_mm])
        shifted_s_d_relative_face_mesh = scaled_demeaned_relative_face_mesh + translation
        
        # Compute the 3D bounding box dimensions of the shifted_s_d_relative_face_mesh
        min_xyz = np.min(shifted_s_d_relative_face_mesh, axis=0)
        max_xyz = np.max(shifted_s_d_relative_face_mesh, axis=0)
        distances = max_xyz - min_xyz
        # print(f"Distances: {distances}")

        # Estimate intrinsics based on width
        intrinsics = np.array([
            [width*1.5, 0, width / 2],
            [0, height*1.9, height / 2],
            [0, 0, 1]
        ])

        # Convert xyz back to uvz
        re_facial_landmarks = np.array([convert_xyz_to_uv_with_intrinsic(intrinsics, x[0], x[1], x[2]) for x in shifted_s_d_relative_face_mesh])
        # re_facial_landmarks = np.array([convert_xyz_to_uv(self.perspective_matrix, x[0], x[1], x[2]) for x in shifted_s_d_relative_face_mesh])
        # re_facial_landmarks = np.array([convert_xyz_to_uv(self.perspective_matrix, x[0], x[1], x[2]) for x in relative_face_mesh])

        # Draw the original facial (DEBUGGING)
        # draw_frame = frame.copy()
        # for (u,v), (nu, nv) in zip(facial_landmarks[:, :2], re_facial_landmarks[:, :2]):
        #     cv2.circle(draw_frame, (int(u * width), int(v * height)), 2, (0, 255, 0), -1)
        #     cv2.circle(draw_frame, (int(nu), int(nv)), 2, (0, 0, 255), -1)
        # cv2.imshow('draw', draw_frame)

        # Compute the average of the 2D eye origins
        face_origin = (positions['eye_origins_2d']['left'] + positions['eye_origins_2d']['right']) / 2
        tf_face_points = shifted_s_d_relative_face_mesh

        # Compute the eye gaze origin in metric space
        eye_g_o = {
            'left': tf_face_points[LEFT_EYE_LANDMARKS],
            'right': tf_face_points[RIGHT_EYE_LANDMARKS]
        }

        # Compute the 3D eye origin
        for k, v in eye_g_o.items():
            eye_g_o[k] = np.mean(v, axis=0)

        # Compute face gaze origin
        face_g_o = (eye_g_o['left'] + eye_g_o['right']) / 2

        return {
            'tf_face_points': tf_face_points,
            'face_origin_3d': face_g_o,
            'face_origin_2d': face_origin,
            'eye_origins_3d': eye_g_o,
            # 'eye_origins_3d': {'left': np.array([0,0,100]), 'right': np.array([0,0,100])},
            'eye_origins_2d': positions['eye_origins_2d']
        }
    
    def compute_pog(self, gaze_origins, gaze_vectors, screen_R, screen_t, screen_width_mm, screen_height_mm, screen_width_px, screen_height_px):
        
        # Perform intersection with plane using gaze origin and vector
        # c for camera, s for screen
        left_pog_mm_c = screen_plane_intersection_2(
            gaze_origins['eye_origins_3d']['left'],
            gaze_vectors['eyes']['vector']['left'],
        )
        right_pog_mm_c = screen_plane_intersection_2(
            gaze_origins['eye_origins_3d']['right'],
            gaze_vectors['eyes']['vector']['right'],
        )

        # Then convert the PoG to screen coordinates
        # Obtain rotation matrix from the Rodrigues vector
        R_matrix, _ = cv2.Rodrigues(screen_R)  # screen_R should be a 3D vector (Rodrigues rotation)
        inv_R_matrix = np.linalg.inv(R_matrix)  # Inverse of the rotation matrix

        # Pad the points from 2 to 3 dimensions
        pad_left_pog_mm_c = np.append(left_pog_mm_c, 0)
        pad_right_pog_mm_c = np.append(right_pog_mm_c, 0)

        # Transform gaze origin and direction to screen coordinates
        left_pog_mm_s = np.dot(inv_R_matrix, (pad_right_pog_mm_c - screen_t.T[0]))
        right_pog_mm_s = np.dot(inv_R_matrix, (pad_left_pog_mm_c - screen_t.T[0]))

        # Convert mm to normalized coordinates
        left_pog_norm = np.array([left_pog_mm_s[0] / screen_width_mm, left_pog_mm_s[1] / screen_height_mm])
        right_pog_norm = np.array([right_pog_mm_s[0] / screen_width_mm, right_pog_mm_s[1] / screen_height_mm])

        # Convert normalized coordinates to pixel coordinates
        left_pog_px = np.array([left_pog_norm[0] * screen_width_px, left_pog_norm[1] * screen_height_px])
        right_pog_px = np.array([right_pog_norm[0] * screen_width_px, right_pog_norm[1] * screen_height_mm])

        return {
            'face_pog_mm_c': (left_pog_mm_c + right_pog_mm_c) / 2,
            'face_pog_mm_s': (left_pog_mm_s + right_pog_mm_s) / 2,
            'face_pog_norm': (left_pog_norm + right_pog_norm) / 2,
            'face_pog_px': (left_pog_px + right_pog_px) / 2,
            'eye': {
                'left_pog_mm_c': left_pog_mm_c,
                'left_pog_mm_s': left_pog_mm_s,
                'left_pog_norm': left_pog_norm,
                'left_pog_px': left_pog_px,
                'right_pog_mm_c': right_pog_mm_c,
                'right_pog_mm_s': right_pog_mm_s,
                'right_pog_norm': right_pog_norm,
                'right_pog_px': right_pog_px
            }
        }

    def step(
            self, 
            frame,
            facial_landmarks, 
            face_rt, 
            face_blendshapes, 
            height, 
            width, 
            intrinsics,
            screen_R=None,
            screen_t=None,
            screen_width_mm=None,
            screen_height_mm=None,
            screen_width_px=None,
            screen_height_px=None,
            tic=None,
            smooth: bool = False
        ):

        if not tic:
            tic = time.perf_counter()

        # If we don't have a perspective matrix, create it
        if type(self.perspective_matrix) == type(None):
            self.perspective_matrix = create_perspective_matrix(aspect_ratio=width/height)
            self.inv_perspective_matrix = np.linalg.inv(self.perspective_matrix)
        
        # Estimate the 2D and 3D position of the eye-center and the face-center
        gaze_origins = self.estimate_2d_3d_eye_face_origins(
            frame,
            facial_landmarks,
            face_rt,
            height,
            width,
            intrinsics
        )

        # Compute the gaze vectors
        if self.gaze_direction_estimation == 'model-based':
            gaze_vectors = self.estimate_gaze_vector_based_on_model_based(
                frame,
                facial_landmarks,
                face_rt,
                width=width,
                height=height
            )
        elif self.gaze_direction_estimation == 'landmark2d':
            gaze_vectors = self.estimate_gaze_vector_based_on_eye_landmarks(
                frame,
                facial_landmarks,
                face_rt,
                width=width,
                height=height
            )
        elif self.gaze_direction_estimation == 'blendshape':
            gaze_vectors = self.estimate_gaze_vector_based_on_eye_blendshapes(
                face_blendshapes,
                face_rt
            )

        # If smooth, apply a moving average filter
        if smooth:
            if self.prior_gaze:
                for k in ['left', 'right']:
                    if not gaze_vectors['eyes']['is_closed'][k]:
                        new_vector = (gaze_vectors['eyes']['vector'][k] + self.prior_gaze['eyes']['vector'][k])
                        gaze_vectors['eyes']['vector'][k] = new_vector / np.linalg.norm(new_vector)
                
                # Update the face gaze vector
                if not gaze_vectors['eyes']['is_closed']['left'] and not gaze_vectors['eyes']['is_closed']['right']:
                    new_vector = (gaze_vectors['eyes']['vector']['left'] + gaze_vectors['eyes']['vector']['right'])
                    gaze_vectors['face'] = new_vector / np.linalg.norm(new_vector)

            self.prior_gaze = gaze_vectors

        # Compute the PoG
        if screen_R is None or screen_t is None or screen_width_mm is None or screen_height_mm is None or screen_width_px is None or screen_height_px is None:
            pog = {
                'face_pog_mm_c': np.array([0,0]),
                'face_pog_mm_s': np.array([0,0]),
                'face_pog_norm': np.array([0,0]),
                'face_pog_px': np.array([0,0]),
                'eye': {
                    'left_pog_mm_c': np.array([0,0]),
                    'left_pog_mm_s': np.array([0,0]),
                    'left_pog_norm': np.array([0,0]),
                    'left_pog_px': np.array([0,0]),
                    'right_pog_mm_c': np.array([0,0]),
                    'right_pog_mm_s': np.array([0,0]),
                    'right_pog_norm': np.array([0,0]),
                    'right_pog_px': np.array([0,0])
                }
            }
        else:
            pog = self.compute_pog(
                gaze_origins,
                gaze_vectors,
                screen_R,
                screen_t,
                screen_width_mm,
                screen_height_mm,
                screen_width_px,
                screen_height_px,
            )

        toc = time.perf_counter()

        # Return the result
        return FLGEResult(
            facial_landmarks=facial_landmarks,
            tf_facial_landmarks=gaze_origins['tf_face_points'],
            face_rt=face_rt,
            face_blendshapes=face_blendshapes,
            face_origin=gaze_origins['face_origin_3d'],
            face_origin_2d=gaze_origins['face_origin_2d'],
            face_gaze=gaze_vectors['face'],
            left=EyeResult(
                is_closed=gaze_vectors['eyes']['is_closed']['left'],
                origin=gaze_origins['eye_origins_3d']['left'],
                origin_2d=gaze_origins['eye_origins_2d']['left'],
                direction=gaze_vectors['eyes']['vector']['left'],
                pog_mm_c=pog['eye']['left_pog_mm_c'],
                pog_mm_s=pog['eye']['left_pog_mm_s'],
                pog_norm=pog['eye']['left_pog_norm'],
                pog_px=pog['eye']['left_pog_px'],
                meta_data={
                    **gaze_vectors['eyes']['meta_data']['left']
                }
            ),
            right=EyeResult(
                is_closed=gaze_vectors['eyes']['is_closed']['right'],
                origin=gaze_origins['eye_origins_3d']['right'],
                origin_2d=gaze_origins['eye_origins_2d']['right'],
                direction=gaze_vectors['eyes']['vector']['right'],
                pog_mm_c=pog['eye']['right_pog_mm_c'],
                pog_mm_s=pog['eye']['right_pog_mm_s'],
                pog_norm=pog['eye']['right_pog_norm'],
                pog_px=pog['eye']['right_pog_px'],
                meta_data={
                    **gaze_vectors['eyes']['meta_data']['right']
                }
            ),
            pog_mm_c=pog['face_pog_mm_c'],
            pog_mm_s=pog['face_pog_mm_s'],
            pog_norm=pog['face_pog_norm'],
            pog_px=pog['face_pog_px'],
            duration=toc - tic
        )
    
    def process_sample(self, frame: np.ndarray, sample: Dict[str, Any], smooth: bool = False) -> FLGEResult:

        # Get the depth and scale
        # frame = cv2.cvtColor(np.moveaxis(sample['image'], 0, -1) * 255, cv2.COLOR_RGB2BGR)
        facial_landmarks = sample['facial_landmarks']
        original_img_size = sample['original_img_size']
        face_rt = sample['facial_rt']

        return self.step(
            frame,
            facial_landmarks,
            face_rt,
            sample['face_blendshapes'],
            original_img_size[0],
            original_img_size[1],
            sample['intrinsics'],
            # sample['screen_R'],
            # sample['screen_t'],
            # sample['screen_width_mm'],
            # sample['screen_height_mm'],
            # sample['screen_width_px'],
            # sample['screen_height_px']
            smooth=smooth
        )
 
    def process_frame(
            self, 
            frame: np.ndarray, 
            intrinsics: np.ndarray, 
            screen_R=None,
            screen_t=None,
            screen_width_mm=None,
            screen_height_mm=None,
            screen_width_px=None,
            screen_height_px=None,
            smooth: bool = False
        ) -> Optional[FLGEResult]:

        # Start a timer
        tic = time.perf_counter()

        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = self.face_landmarker.detect(mp_image)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            return None
        
        # Extract information fro the results
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
        face_rt = detection_results.facial_transformation_matrixes[0]
        face_blendshapes = np.array([bs.score for bs in detection_results.face_blendshapes[0]])
        
        # Perform step
        return self.step(
            frame.astype(np.uint8),
            face_landmarks,
            face_rt,
            face_blendshapes,
            frame.shape[0],
            frame.shape[1],
            intrinsics,
            screen_R=screen_R,
            screen_t=screen_t,
            screen_width_mm=screen_width_mm,
            screen_height_mm=screen_height_mm,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            tic=tic,
            smooth=smooth
        ) 