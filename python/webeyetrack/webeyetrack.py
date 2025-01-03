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

from .model_based import create_perspective_matrix, estimate_2d_3d_eye_face_origins, estimate_gaze_vector_based_on_model_based, estimate_gaze_vector_based_on_eye_landmarks, estimate_gaze_vector_based_on_eye_blendshapes, compute_pog
from .data_protocols import GazeResult, EyeResult
from .constants import *

class WebEyeTrack():

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
        gaze_origins = estimate_2d_3d_eye_face_origins(
            self.perspective_matrix,
            facial_landmarks,
            face_rt,
            height,
            width,
            intrinsics
        )

        # Compute the gaze vectors
        if self.gaze_direction_estimation == 'model-based':
            gaze_vectors = estimate_gaze_vector_based_on_model_based(
                self.eyeball_centers,
                self.eyeball_radius,
                self.perspective_matrix,
                self.inv_perspective_matrix,
                facial_landmarks,
                face_rt,
                width=width,
                height=height,
                ear_threshold=self.ear_threshold
            )
        elif self.gaze_direction_estimation == 'landmark2d':
            gaze_vectors = estimate_gaze_vector_based_on_eye_landmarks(
                facial_landmarks,
                face_rt,
                width=width,
                height=height
            )
        elif self.gaze_direction_estimation == 'blendshape':
            gaze_vectors = estimate_gaze_vector_based_on_eye_blendshapes(
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
            pog = compute_pog(
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
        return GazeResult(
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
    
    def process_sample(self, frame: np.ndarray, sample: Dict[str, Any], smooth: bool = False) -> GazeResult:

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
        ) -> Optional[GazeResult]:

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