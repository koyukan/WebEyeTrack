import time
from typing import Dict, Any, Literal, Optional
import math

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from webeyetrack.datasets.utils import screen_plane_intersection
from ..vis import draw_axis
from ..core import pitch_yaw_to_gaze_vector, rotation_matrix_to_euler_angles
from ..data_protocols import FLGEResult, EyeResult

LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]
LEFT_BLENDSHAPES = [14, 16, 18, 12]
RIGHT_BLENDSHAPES = [13, 15, 17, 11]
REAL_WORLD_IPD = 6.3 # Inter-pupilary distance (cm)
HFOV = 100
VFOV = 90

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

RIGHT_IRIS_LANDMARKS = [468, 470, 469, 472, 471] # center, top, right, botton, left
LEFT_IRIS_LANDMARKS = [473, 475, 474, 477, 476] # center, top, right, botton, left

# EYE_PADDING_HEIGHT = 0.1
EYE_PADDING_WIDTH = 0.3
EYE_HEIGHT_RATIO = 0.7

# Position of eyeball center based on canonical coordinate system
LEFT_EYEBALL_CENTER = np.array([3.0278, -2.7526, 2.7234]) * 10 # X, Y, Z
RIGHT_EYEBALL_CENTER = np.array([-3.0278, -2.7526, 2.7234]) * 10 # X, Y, Z

# Average radius of an eyeball in cm
EYEBALL_RADIUS = 1.15

def compute_2d_origin(points):
    (cx, cy), radius = cv2.minEnclosingCircle(points.astype(np.float32))
    center = np.array([cx, cy], dtype=np.int32)
    return center

# Facial Landmark Gaze Estimation
class FLGE():

    def __init__(self, model_asset_path: str, gaze_direction_estimation: Literal['landmark', 'blendshape'] = 'landmark'):

        # Saving options
        self.gaze_direction_estimation = gaze_direction_estimation

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

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
            cv2.circle(frame, (int(old_iris_center[0]), int(old_iris_center[1])), 2, (0, 0, 255), -1)
            shifted_iris_center = old_iris_center + np.array([int(x3), int(y3)])
            cv2.circle(frame, (int(shifted_iris_center[0]), int(shifted_iris_center[1])), 2, (0, 255, 0), -1)
            cv2.line(frame, (int(old_iris_center[0]), int(old_iris_center[1])), (int(shifted_iris_center[0]), int(shifted_iris_center[1])), (0, 255, 0), 1)

            # Shifting the eye_center by the headpose
            eye_center = eye_center + np.array([int(x3), int(y3)])

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
        elif 'right' in gaze_vectors:
            face_gaze_vector = gaze_vectors['right']
            gaze_vectors['left'] = np.array([0,0,1])
        else:
            face_gaze_vector = np.array([0,0,1])
            gaze_vectors['left'] = np.array([0,0,1])
            gaze_vectors['right'] = np.array([0,0,1])

        return {
            'face': face_gaze_vector,
            'eyes': {
                'is_closed': eye_closed,
                'vector': gaze_vectors
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
            }            
        }

    def estimate_2d_3d_eye_face_origins(self, facial_landmarks, face_rt, height, width, intrinsics):
        
        # First, compute the inter-pupillary distance
        positions, distances = self.estimate_inter_pupillary_distance_2d(
            facial_landmarks, 
            height, 
            width
        )

        # Estimate the scale
        metric_scale = REAL_WORLD_IPD / distances['canonical_ipd_3d']

        # Estimate the depth
        focal_length_pixels = width / 2
        # depth_cm = (focal_length_pixels * REAL_WORLD_IPD * np.cos(theta)) / distances['image_ipd']
        depth_cm = (focal_length_pixels * REAL_WORLD_IPD) / distances['image_ipd']
        depth_mm = depth_cm * 10 * 1.718

        # Get the transformation matrix
        # Invert the y and z axis
        transform = face_rt.copy()
        transform = np.diag([-1, 1, 1, 1]) @ transform

        # Apply the metric scaling of the face points
        tf_face_points = np.copy(facial_landmarks[:, :3])
        tf_face_points *= metric_scale

        # Compute the average of the 2D eye origins
        face_origin = (positions['eye_origins_2d']['left'] + positions['eye_origins_2d']['right']) / 2

        # Compute the position based on the 2D xy and the depth
        pixel_coords = np.array([face_origin[0], face_origin[1], 1])
        K = intrinsics
        K_inv = np.linalg.inv(K)

        # Back project the pixel coordinates to the 3D coordinates
        translation = depth_mm * K_inv @ pixel_coords

        # Apply the translation to the face points
        tf_face_points += translation

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
            'face_origin_3d': face_g_o,
            'face_origin_2d': face_origin,
            'eye_origins_3d': eye_g_o,
            'eye_origins_2d': positions['eye_origins_2d']
        }
    
    def compute_pog(self, gaze_origins, gaze_vectors, screen_R, screen_t, screen_width_mm, screen_height_mm, screen_width_px, screen_height_px):
        
        # Perform intersection with plane using gaze origin and vector
        left_pog_mm = screen_plane_intersection(
            gaze_origins['eye_origins_3d']['left'],
            gaze_vectors['eyes']['vector']['left'],
            screen_R,
            screen_t
        )
        right_pog_mm = screen_plane_intersection(
            gaze_origins['eye_origins_3d']['right'],
            gaze_vectors['eyes']['vector']['right'],
            screen_R,
            screen_t
        )

        # Convert mm to normalized coordinates
        left_pog_norm = np.array([left_pog_mm[0] / screen_width_mm, left_pog_mm[1] / screen_height_mm])
        right_pog_norm = np.array([right_pog_mm[0] / screen_width_mm, right_pog_mm[1] / screen_height_mm])

        # Convert normalized coordinates to pixel coordinates
        left_pog_px = np.array([left_pog_norm[0] * screen_width_px, left_pog_norm[1] * screen_height_px])
        right_pog_px = np.array([right_pog_norm[0] * screen_width_px, right_pog_norm[1] * screen_height_mm])

        return {
            'face_pog_mm': (left_pog_mm + right_pog_mm) / 2,
            'face_pog_px': (left_pog_px + right_pog_px) / 2,
            'eye': {
                'left_pog_mm': left_pog_mm,
                'left_pog_px': left_pog_px,
                'right_pog_mm': right_pog_mm,
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
            tic=None
        ):

        if not tic:
            tic = time.perf_counter()
        
        # Estimate the 2D and 3D position of the eye-center and the face-center
        gaze_origins = self.estimate_2d_3d_eye_face_origins(
            facial_landmarks,
            face_rt,
            height,
            width,
            intrinsics
        )

        # Compute the gaze vectors
        if self.gaze_direction_estimation == 'landmark':
            gaze_vectors = self.estimate_gaze_vector_based_on_eye_landmarks(
                frame,
                facial_landmarks,
                face_rt,
                width,
                height
            )
        elif self.gaze_direction_estimation == 'blendshape':
            gaze_vectors = self.estimate_gaze_vector_based_on_eye_blendshapes(
                face_blendshapes,
                face_rt
            )

        # Compute the PoG
        if screen_R is None or screen_t is None or screen_width_mm is None or screen_height_mm is None or screen_width_px is None or screen_height_px is None:
            pog = {
                'face_pog_mm': np.array([0,0]),
                'face_pog_px': np.array([0,0]),
                'eye': {
                    'left_pog_mm': np.array([0,0]),
                    'left_pog_px': np.array([0,0]),
                    'right_pog_mm': np.array([0,0]),
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
                pog_px=pog['eye']['left_pog_px'],
                pog_mm=pog['eye']['left_pog_mm']
            ),
            right=EyeResult(
                is_closed=gaze_vectors['eyes']['is_closed']['right'],
                origin=gaze_origins['eye_origins_3d']['right'],
                origin_2d=gaze_origins['eye_origins_2d']['right'],
                direction=gaze_vectors['eyes']['vector']['right'],
                pog_px=pog['eye']['right_pog_px'],
                pog_mm=pog['eye']['right_pog_mm']
            ),
            pog_px=pog['face_pog_px'],
            pog_mm=pog['face_pog_mm'],
            duration=toc - tic
        )
    
    def process_sample(self, sample: Dict[str, Any]) -> FLGEResult:

        # Get the depth and scale
        frame = cv2.cvtColor(np.moveaxis(sample['image'], 0, -1) * 255, cv2.COLOR_RGB2BGR)
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
            sample['screen_R'],
            sample['screen_t'],
            sample['screen_width_mm'],
            sample['screen_height_mm'],
            sample['screen_width_px'],
            sample['screen_height_px']
        )
 
    def process_frame(self, frame: np.ndarray, intrinsics: np.ndarray) -> Optional[FLGEResult]:

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
            tic=tic
        ) 