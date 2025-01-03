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
            frame_height: int,
            frame_width: int,
            intrinsics: np.ndarray,
            screen_R: np.ndarray,
            screen_t: np.ndarray,
            screen_width_mm: float,
            screen_height_mm: float,
            screen_width_px: int,
            screen_height_px: int,
            eyeball_centers: Tuple[np.ndarray, np.ndarray] = EYEBALL_DEFAULT,
            eyeball_radius: float = EYEBALL_RADIUS,
            ear_threshold: float = 0.1,
        ):

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
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.eyeball_centers = eyeball_centers
        self.eyeball_radius = eyeball_radius
        self.ear_threshold = ear_threshold
        self.intrinsics = intrinsics
        self.screen_R = screen_R
        self.screen_t = screen_t
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px

        # Gaze filter
        self.prior_gaze = None
        self.prior_depth = None

    def calibrate(self, samples):

        def loss_fn(params):
            # The loss function is the sum of the squared differences between the esimated PoG and the ground truth PoG
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
        result = gp_minimize(
            func=loss_fn,
            dimensions=dimensions,
            x0=x0,
            n_calls=2,
            random_state=42
        )

        print(result)
        
    def step(
            self, 
            facial_landmarks, 
            face_rt, 
            face_blendshapes, 
        ):

        tic = time.perf_counter()

        # If we don't have a perspective matrix, create it
        if type(self.perspective_matrix) == type(None):
            self.perspective_matrix = create_perspective_matrix(aspect_ratio=self.frame_width/self.frame_height)
            self.inv_perspective_matrix = np.linalg.inv(self.perspective_matrix)
        
        # Estimate the 2D and 3D position of the eye-center and the face-center
        gaze_origins = estimate_2d_3d_eye_face_origins(
            self.perspective_matrix,
            facial_landmarks,
            face_rt,
            self.frame_height,
            self.frame_width,
            self.intrinsics
        )

        # Compute the gaze vectors
        gaze_vectors = estimate_gaze_vector_based_on_model_based(
            self.eyeball_centers,
            self.eyeball_radius,
            self.perspective_matrix,
            self.inv_perspective_matrix,
            facial_landmarks,
            face_rt,
            width=self.frame_width,
            height=self.frame_height,
            ear_threshold=self.ear_threshold
        )

        # Compute the PoG
        pog = compute_pog(
            gaze_origins,
            gaze_vectors,
            self.screen_R,
            self.screen_t,
            self.screen_width_mm,
            self.screen_height_mm,
            self.screen_width_px,
            self.screen_height_px
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
    
    def process_sample(self, frame: np.ndarray, sample: Dict[str, Any]) -> GazeResult:

        # Get the depth and scale
        # frame = cv2.cvtColor(np.moveaxis(sample['image'], 0, -1) * 255, cv2.COLOR_RGB2BGR)
        facial_landmarks = sample['facial_landmarks']
        face_rt = sample['facial_rt']

        return self.step(
            frame,
            facial_landmarks,
            face_rt,
            sample['face_blendshapes'],
        )
 
    def process_frame(
            self, 
            frame: np.ndarray, 
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
            face_landmarks,
            face_rt,
            face_blendshapes
        )