from typing import Dict, Any

import numpy as np
import cv2

from webeyetrack.datasets.utils import screen_plane_intersection

LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]
LEFT_BLENDSHAPES = [14, 16, 18, 12]
RIGHT_BLENDSHAPES = [13, 15, 17, 11]
REAL_WORLD_IPD = 6.3 # Inter-pupilary distance (cm)
HFOV = 100
VFOV = 90

def compute_2d_origin(points):
    (cx, cy), radius = cv2.minEnclosingCircle(points.astype(np.float32))

    center = np.array([cx, cy], dtype=np.int32)
    return center

# Facial Landmark Gaze Estimation
class FLGE():

    def estimate_inter_pupillary_distance_2d(self, sample):
        data_2d_pairs = {
            'left': sample['facial_landmarks_2d'][LEFT_EYE_LANDMARKS],
            'right': sample['facial_landmarks_2d'][RIGHT_EYE_LANDMARKS]
        }
        data_3d_pairs = {
            'left': sample['facial_landmarks'][LEFT_EYE_LANDMARKS][:, :3],
            'right': sample['facial_landmarks'][RIGHT_EYE_LANDMARKS][:, :3]
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
    
    def estimate_depth_and_scale(self, sample):
        # First, compute the inter-pupillary distance
        distances = self.estimate_inter_pupillary_distance_2d(sample)

        # Correct the image_ipd with the facial rotation
        face_rt = sample['facial_rt']
        # theta = np.arccos((np.trace(face_rt[:3, :3]) - 1) / 2)

        # Estimate the scale
        metric_scale = REAL_WORLD_IPD / distances['canonical_ipd_3d']

        # Estimate the depth
        # Assuming the focal length is half the image width
        focal_length_pixels = sample['original_img_size'][1] / 2
        # depth_cm = (focal_length_pixels * REAL_WORLD_IPD * np.cos(theta)) / distances['image_ipd']
        depth_cm = (focal_length_pixels * REAL_WORLD_IPD) / distances['image_ipd']
        depth_mm = depth_cm * 10 * 1.718

        return {
            'depth': depth_mm,
            'metric_scale': metric_scale,
            **distances
        }
    
    def compute_gaze_origin_and_direction(self, sample, data):
        metric_scale = data['metric_scale']
        depth_mm = data['depth']
        gaze_2d_origins = data['2d_eye_origins']

        # Get the transformation matrix
        transform = sample['facial_rt']

        # Invert the y and z axis
        transform = np.diag([-1, 1, 1, 1]) @ transform

        # Apply the metric scaling of the face points
        face_points = np.copy(sample['facial_landmarks'][:, :3])
        face_points *= metric_scale

        # Compute the average of the 2D eye origins
        eye_origin = (gaze_2d_origins['left'] + gaze_2d_origins['right']) / 2

        # Compute the position based on the 2D xy and the depth
        pixel_coords = np.array([eye_origin[0], eye_origin[1], 1])
        K = sample['intrinsics']
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
            blendshapes = sample['face_blendshapes']
            look_in, look_out, look_up, look_down = ([blendshapes[i].score for i in value])
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
            'face_gaze_vector': gaze_vector
        }
    
    def compute_pog(self, sample, data):
        
        # Perform intersection with plane using gaze origin and vector
        left_pog_mm = screen_plane_intersection(
            data['gaze_origins']['left'],
            data['gaze_vectors']['left'],
            sample['screen_R'],
            sample['screen_t']
        )
        right_pog_mm = screen_plane_intersection(
            data['gaze_origins']['right'],
            data['gaze_vectors']['right'],
            sample['screen_R'],
            sample['screen_t']
        )

        # Convert mm to normalized coordinates
        left_pog_norm = np.array([left_pog_mm[0] / sample['screen_width_mm'], left_pog_mm[1] / sample['screen_height_mm']])
        right_pog_norm = np.array([right_pog_mm[0] / sample['screen_width_mm'], right_pog_mm[1] / sample['screen_height_mm']])

        # Convert normalized coordinates to pixel coordinates
        left_pog_px = np.array([left_pog_norm[0] * sample['screen_width_px'], left_pog_norm[1] * sample['screen_height_px']])
        right_pog_px = np.array([right_pog_norm[0] * sample['screen_width_px'], right_pog_norm[1] * sample['screen_height_px']])

        return {
            'pog_mm': (left_pog_mm + right_pog_mm) / 2,
            'pog_px': (left_pog_px + right_pog_px) / 2
        }

    def process_sample(self, sample: Dict[str, Any]):

        # Get the depth and scale
        data = self.estimate_depth_and_scale(sample)

        # Compute the metric transformation matrix
        data2 = self.compute_gaze_origin_and_direction(sample, data)

        # Compute the PoG
        data3 = self.compute_pog(sample, data2)

        return {
            **data,
            **data2,
            **data3
        }
