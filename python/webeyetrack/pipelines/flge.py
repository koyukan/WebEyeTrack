from typing import Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from webeyetrack.datasets.utils import screen_plane_intersection
from ..vis import draw_axis

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

# Average radius of an eyeball in cm
EYEBALL_RADIUS = 1.15

def compute_2d_origin(points):
    (cx, cy), radius = cv2.minEnclosingCircle(points.astype(np.float32))

    center = np.array([cx, cy], dtype=np.int32)
    return center

@dataclass
class EyeResult:
    is_closed: bool
    origin: np.ndarray # X, Y, Z
    direction: np.ndarray # Pitch, Yaw
    pog_px: np.ndarray
    pog_mm: np.ndarray

@dataclass
class FLGEResult:
    face_origin: np.ndarray # X, Y, Z
    face_origin_2d: np.ndarray # X, Y
    face_gaze: np.ndarray # Pitch, Yaw
    left: EyeResult
    right: EyeResult
    pog_px: np.ndarray
    pog_mm: np.ndarray

# Facial Landmark Gaze Estimation
class FLGE():

    def __init__(self, model_asset_path: str):
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

        return {
            '2d_eye_origins': {
                'left': l,
                'right': r
            },
            '3d_eye_origins': {
                'left': origins_3d['left'],
                'right': origins_3d['right']
            },
            'canonical_ipd_3d': canonical_ipd,
            'image_ipd': image_ipd
        }
    
    def estimate_depth_and_scale(self, facial_landmarks, face_rt, height, width):
        # First, compute the inter-pupillary distance
        distances = self.estimate_inter_pupillary_distance_2d(facial_landmarks, height, width)

        # Correct the image_ipd with the facial rotation
        theta = np.arccos((np.trace(face_rt[:3, :3]) - 1) / 2)

        # Estimate the scale
        metric_scale = REAL_WORLD_IPD / distances['canonical_ipd_3d']

        # Estimate the depth
        # Assuming the focal length is half the image width
        # focal_length_pixels = sample['original_img_size'][1] / 2
        focal_length_pixels = width / 2
        # depth_cm = (focal_length_pixels * REAL_WORLD_IPD * np.cos(theta)) / distances['image_ipd']
        depth_cm = (focal_length_pixels * REAL_WORLD_IPD) / distances['image_ipd']
        depth_mm = depth_cm * 10 * 1.718

        return {
            'depth': depth_mm,
            'metric_scale': metric_scale,
            **distances
        }
    
    def compute_gaze_origin_and_direction(self, 
            facial_landmarks, 
            facial_rt, 
            face_blendshapes,
            metric_scale, 
            depth_mm, 
            gaze_2d_origins,
            intrinsics
        ):

        # Get the transformation matrix
        transform = facial_rt

        # Invert the y and z axis
        transform = np.diag([-1, 1, 1, 1]) @ transform

        # Apply the metric scaling of the face points
        face_points = np.copy(facial_landmarks[:, :3])
        face_points *= metric_scale

        # Compute the average of the 2D eye origins
        eye_origin = (gaze_2d_origins['left'] + gaze_2d_origins['right']) / 2

        # Compute the position based on the 2D xy and the depth
        pixel_coords = np.array([eye_origin[0], eye_origin[1], 1])
        K = intrinsics
        K_inv = np.linalg.inv(K)

        # Back project the pixel coordinates to the 3D coordinates
        translation = depth_mm * K_inv @ pixel_coords

        # Apply the translation to the face points
        face_points += translation

        # Compute the gaz
        gaze_origins = {
            'left': face_points[LEFT_EYE_LANDMARKS],
            'right': face_points[RIGHT_EYE_LANDMARKS]
        }

        # Compute the 3D eye origin
        for k, v in gaze_origins.items():
            gaze_origins[k] = np.mean(v, axis=0)

        # Compute average gaze origin
        gaze_origin = (gaze_origins['left'] + gaze_origins['right']) / 2

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
        gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
        gaze_vector /= np.linalg.norm(gaze_vector)

        return {
            'gaze_origins': gaze_origins,
            'gaze_directions': gaze_directions,
            'gaze_vectors': gaze_vectors,
            'gaze_origin': gaze_origin,
            'face_gaze_vector': gaze_vector,
            'gaze_origin_2d': eye_origin
        }
    
    def estimate_gaze_vector(self, face_landmarks_proto, frame, render=False):
        
        # Compute the bbox by using the edges of the each eyes
        face_landmarks_all = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
        left_2d_eye_px = face_landmarks_all[LEFT_EYEAREA_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyelid_px = face_landmarks_all[LEFT_EYELID_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_iris_px = face_landmarks_all[LEFT_IRIS_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyearea_total_px = face_landmarks_all[LEFT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyelid_total_px = face_landmarks_all[LEFT_EYELID_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        
        right_2d_eye_px = face_landmarks_all[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyelid_px = face_landmarks_all[RIGHT_EYELID_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_iris_px = face_landmarks_all[RIGHT_IRIS_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyearea_total_px = face_landmarks_all[RIGHT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyelid_total_px = face_landmarks_all[RIGHT_EYELID_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])

        # 3D
        left_eye_fl = face_landmarks_all[LEFT_EYELID_LANDMARKS, :3]
        right_eye_fl = face_landmarks_all[RIGHT_EYELID_LANDMARKS, :3]

        left_landmarks = [
            left_2d_eye_px, 
            left_2d_eyelid_px,
            left_2d_eyearea_total_px,
            left_2d_eyelid_total_px
        ]
        right_landmarks = [
            right_2d_eye_px, 
            right_2d_eyelid_px,
            right_2d_eyearea_total_px,
            right_2d_eyelid_total_px
        ]

        eye_closed = {}
        eye_images = {}
        gaze_vectors = {}
        gaze_origins = {}
        for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
            centroid = np.mean(eye, axis=0)
            width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
            height = width * EYE_HEIGHT_RATIO

            # Determine if closed by the eyelid
            eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
            eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
            is_closed = False

            # Determine if the eye is closed by the ratio of the height based on the width
            if eyelid_height / eyelid_width < 0.05:
                is_closed = True

            if width == 0 or height == 0:
                continue

            # Crop the eye
            eye_image = frame[
                int(centroid[1] - height/2):int(centroid[1] + height/2),
                int(centroid[0] - width/2):int(centroid[0] + width/2)
            ]

            if eye_image.shape[0] == 0 or eye_image.shape[1] == 0:
                continue

            # Resize the eye
            # eye_image = imutils.resize(eye_image, width=400)
            original_height, original_width = eye_image.shape[:2]
            new_width, new_height = 400, int(400*EYE_HEIGHT_RATIO)
            eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            eye_images[i] = eye_image

            # Draw the outline of the eyearea
            if render:
                shifted_eyearea_px = eyearea - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
                prior_px = None
                for px in shifted_eyearea_px:
                    resized_px = px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                    if prior_px is not None:
                        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (0, 255, 0), 1)
                    prior_px = resized_px
                # Draw the last line to close the loop
                resized_first_px = shifted_eyearea_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (0, 255, 0), 1)

                # Draw the outline of the eyelid
                shifted_eyelid_px = eyelid_total - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
                prior_px = None
                for px in shifted_eyelid_px:
                    resized_px = px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                    if prior_px is not None:
                        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
                    prior_px = resized_px
                # Draw the last line to close the loop
                resized_first_px = shifted_eyelid_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

            # Draw if the eye is closed on the top left corner
            eye_closed[i] = is_closed
            if is_closed:
                if render:
                    cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # Shift the IRIS landmarks to the cropped eye
            iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
            shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
            iris_center = shifted_iris_px[0]
            eye_center = np.array([width/2, height/2])

            # Based on the direction and magnitude of the line, compute the gaze direction
            # Compute 2D vector from eyeball center to iris center
            # gaze_vector_2d = shifted_iris_px[0] - iris_px[0]
            gaze_vector_2d = iris_center - eye_center
            # gaze_vector_2d = np.array([0,0])

            # Estimate the depth (Z) based on the 2D vector length
            # z_depth = EYEBALL_RADIUS / np.linalg.norm(gaze_vector_2d)
            # import pdb; pdb.set_trace()
            z_depth = 2.0

            # Compute yaw (horizontal rotation)
            yaw = np.arctan2(gaze_vector_2d[0] * 0.1, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Compute pitch (vertical rotation)
            pitch = np.arctan2(gaze_vector_2d[1] * 0.1, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Store pitch and yaw
            gaze_vectors[i] = np.array([pitch, yaw])

            # Compute 3D gaze origin
            eye_fl = left_eye_fl if i == 'left' else right_eye_fl
            gaze_origin = np.mean(eye_fl, axis=0)
            gaze_origins[i] = gaze_origin

            # Draw the center dot
            if render:
                for iris_px in shifted_iris_px:
                    resized_iris_px = iris_px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                    cv2.circle(eye_image, tuple(resized_iris_px.astype(int)), 3, (0, 0, 255), -1)

                # Draw the centroid of the eyeball
                cv2.circle(eye_image, (int(new_width/2), int(new_height/2)), 3, (255, 0, 0), -1)

                # Compute the line between the iris center and the centroid
                new_shifted_iris_px_center = shifted_iris_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                cv2.line(eye_image, (int(new_width/2), int(new_height/2)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 0), 2)

        return {
            'eye_closed': eye_closed,
            'eye_images': eye_images,
            'gaze_vectors': gaze_vectors,
            'gaze_origins': gaze_origins
        }
    
    def compute_pog(self, gaze_origins, gaze_vectors, screen_R, screen_t, screen_width_mm, screen_height_mm, screen_width_px, screen_height_px):
        
        # Perform intersection with plane using gaze origin and vector
        left_pog_mm = screen_plane_intersection(
            gaze_origins['left'],
            gaze_vectors['left'],
            screen_R,
            screen_t
        )
        right_pog_mm = screen_plane_intersection(
            gaze_origins['right'],
            gaze_vectors['right'],
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
            'pog_mm': (left_pog_mm + right_pog_mm) / 2,
            'pog_px': (left_pog_px + right_pog_px) / 2
        }

    def process_sample(self, sample: Dict[str, Any]) -> FLGEResult:

        # Get the depth and scale
        facial_landmarks = sample['facial_landmarks']
        original_img_size = sample['original_img_size']
        face_rt = sample['facial_rt']
        data = self.estimate_depth_and_scale(
            facial_landmarks, 
            face_rt,
            original_img_size[0], 
            original_img_size[1]
        )

        # Compute the metric transformation matrix
        data2 = self.compute_gaze_origin_and_direction(
            facial_landmarks,
            face_rt,
            sample['face_blendshapes'],
            data['metric_scale'],
            data['depth'],
            data['2d_eye_origins'],
            sample['intrinsics']
        )

        # Compute the PoG
        data3 = self.compute_pog(
            data2['gaze_origins'],
            data2['gaze_vectors'],
            sample['screen_R'],
            sample['screen_t'],
            sample['screen_width_mm'],
            sample['screen_height_mm'],
            sample['screen_width_px'],
            sample['screen_height_px']
        )

        # Return the result
        return FLGEResult(
            face_origin=data2['gaze_origin'],
            face_origin_2d=data2['gaze_origin_2d'],
            face_gaze=data2['face_gaze_vector'],
            left=EyeResult(
                is_closed=False,
                origin=data2['gaze_origins']['left'],
                direction=data2['gaze_vectors']['left'],
                pog_px=data3['pog_px'],
                pog_mm=data3['pog_mm']
            ),
            right=EyeResult(
                is_closed=False,
                origin=data2['gaze_origins']['right'],
                direction=data2['gaze_vectors']['right'],
                pog_px=data3['pog_px'],
                pog_mm=data3['pog_mm']
            ),
            pog_px=data3['pog_px'],
            pog_mm=data3['pog_mm']
        )
    
    def process_frame(self, frame: np.ndarray, render: bool = False) -> Dict[str, Any]:

        # Return container
        output = {}
        
        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = self.face_landmarker.detect(mp_image)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            return output
        
        # Estimate the gaze
        gaze_estimation = self.estimate_gaze_vector(face_landmarks_proto, frame, render=render)
        output.update(gaze_estimation)

        # If render is True, render the gaze visualization
        if render:
            self.render(output, frame)

        return output
    
    def render(self, output: Dict[str, Any], frame: np.ndarray):
        h, w = frame.shape[:2]

        # Assuming eye_images contains the left and right eye images (both resized to 400x280)
        # Draw the 3D Gaze vector
        for eye in ['left', 'right']:
            if eye not in output['gaze_origins']:
                continue
            gaze_origin = output['gaze_origins'][eye]
            gaze_origin = output['gaze_origins'][eye] * np.array([w, h, 1])
            gaze_pitch_yaw = output['gaze_vectors'][eye]
            frame = draw_axis(frame, gaze_pitch_yaw[1], gaze_pitch_yaw[0], 0, tdx=gaze_origin[0], tdy=gaze_origin[1], size=100)

        if 'left' in output['eye_images'] and 'right' in output['eye_images']:
            left_eye_image = output['eye_images']['left']
            right_eye_image = output['eye_images']['right']

            # Concatenate the eye images horizontally
            eyes_combined = cv2.hconcat([right_eye_image, left_eye_image])

            # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
            eyes_combined_resized = cv2.resize(eyes_combined, (frame.shape[1], eyes_combined.shape[0]))

            # Concatenate the combined eyes image vertically with the frame
            final_image = cv2.vconcat([frame, eyes_combined_resized])

            # Display the final concatenated image
            # cv2.imshow('Gaze Visualization', final_image)
            output['gaze_visualization'] = final_image