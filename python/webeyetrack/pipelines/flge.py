from typing import Dict, Any

import numpy as np
import cv2

LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]
REAL_WORLD_IPD = 6.3 # Inter-pupilary distance

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
            'canonical_ipd_3d': canonical_ipd,
            'image_ipd': image_ipd
        }

    def process_sample(self, sample: Dict[str, Any]):

        # First, compute the inter-pupillary distance
        distances = self.estimate_inter_pupillary_distance_2d(sample)

        # Estimate the depth and scale
        metric_scale = REAL_WORLD_IPD / distances['canonical_ipd_3d']
        depth = (sample['original_img_size'][1] / 2) * REAL_WORLD_IPD / distances['image_ipd']

        return {
            'scale': metric_scale,
            'depth': depth
        }