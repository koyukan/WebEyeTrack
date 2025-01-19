import time
from typing import Dict, Any, Literal, Optional, Tuple
from collections import deque

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skopt import gp_minimize
from skopt.space import Real

from .model_based import (
    create_perspective_matrix, 
    face_reconstruction,
    compute_ear,
    estimate_face_width,
    estimate_gaze_origins,
    estimate_gaze_vector_based_on_eye_blendshapes, 
    compute_pog
)
from .vis import TimeSeriesOscilloscope
from .data_protocols import GazeResult, EyeResult
from .constants import *

PARAMETER_LIST = [
    'frame_height',
    'frame_width',
    'face_width_cm',
    'intrinsics',
    'screen_R',
    'screen_t',
    'screen_width_mm',
    'screen_height_mm',
    'screen_width_px',
    'screen_height_px',
]

class WebEyeTrack():

    def __init__(
            self, 
            model_asset_path: str, 
            frame_height: Optional[int] = None,
            frame_width: Optional[int] = None,
            intrinsics: Optional[np.ndarray] = None,
            face_width_cm: Optional[float] = None,
            screen_R: Optional[np.ndarray] = None,
            screen_t: Optional[np.ndarray] = None,
            screen_width_mm: Optional[float] = None,
            screen_height_mm: Optional[float] = None,
            screen_width_px: Optional[int] = None,
            screen_height_px: Optional[int] = None,
            eyeball_centers: Tuple[np.ndarray, np.ndarray] = EYEBALL_DEFAULT,
            eyeball_radius: float = EYEBALL_RADIUS,
            ear_threshold: float = 0.2,
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
        self.face_width_cm = face_width_cm
        self.intrinsics = intrinsics
        self.screen_R = screen_R
        self.screen_t = screen_t
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px

        # State variables
        self.prior_gaze = None
        self.prior_depth = None

    def config(self, **kwargs):
        for key, value in kwargs.items():
            if key in PARAMETER_LIST:
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid key: {key} in {PARAMETER_LIST}')

    def calibrate(self, samples):

        def loss_fn(params):
            # The loss function is the sum of the squared differences between the esimated PoG and the ground truth PoG
            # For all samples, compute the error
            errors = []
            for sample in samples:

                # Use the current parameters to compute the PoG
                # eyeball_centers = (np.array([params[1], params[2], params[3]]), np.array([-params[1], params[2], params[3]]))
                eyeball_centers = (np.array([params[0], params[1], params[2]]), np.array([-params[0], params[1], params[2]]))
                # eyeball_radius = params[0]
                results = self.process_sample(sample['image'], sample, eyeball_centers=eyeball_centers)

                # Compute the error
                error = np.linalg.norm(results.pog_mm_s - sample['pog_mm'].reshape((2)))
                errors.append(error)

            return sum(errors)

        dimensions = [
            # Real(15/2, 25/2, name='eyeball_radius'),
            Real(EYEBALL_X/2, EYEBALL_X*2, name='eyeball_x'),
            Real(EYEBALL_Y/2, EYEBALL_Y*2, name='eyeball_y'),
            Real(EYEBALL_Z/2, EYEBALL_Z*2, name='eyeball_z')
        ]

        # Initial guess for the parameters
        # x0 = [EYEBALL_RADIUS, EYEBALL_X, EYEBALL_Y, EYEBALL_Z]
        x0 = [EYEBALL_X, EYEBALL_Y, EYEBALL_Z]

        # Perform the optimization
        result = gp_minimize(
            func=loss_fn,
            dimensions=dimensions,
            x0=x0,
            n_calls=11,
            random_state=42
        )

        print(result)
        import pdb; pdb.set_trace() # TODO
        
    def step(
            self, 
            facial_landmarks, 
            face_rt, 
            face_blendshapes, 
        ):

        tic = time.perf_counter()

        # Convert norm uv to pixel space
        facial_landmarks_px = facial_landmarks[:, :2] * np.array([self.frame_width, self.frame_height])

        if self.face_width_cm is None:
            self.face_width_cm = estimate_face_width(facial_landmarks, face_rt)

        # If we don't have a perspective matrix, create it
        if type(self.perspective_matrix) == type(None):
            self.perspective_matrix = create_perspective_matrix(aspect_ratio=self.frame_width/self.frame_height)
            self.inv_perspective_matrix = np.linalg.inv(self.perspective_matrix)

        # Perform 3D face reconstruction and determine the pose in 3D centimeters
        metric_transform, metric_face = face_reconstruction(
            perspective_matrix=self.perspective_matrix,
            face_landmarks=facial_landmarks,
            face_width_cm=self.face_width_cm,
            face_rt=face_rt,
            K=self.intrinsics,
            frame_height=self.frame_height,
            frame_width=self.frame_width
        )

        # Obtain the gaze origins based on the metric face pts
        gaze_origins = estimate_gaze_origins(
            face_landmarks_3d=metric_face,
            face_landmarks=facial_landmarks_px,
        )

        # Estimate the gaze based on the face blendshapes
        gaze_vectors = estimate_gaze_vector_based_on_eye_blendshapes(
            face_blendshapes=face_blendshapes,
            face_rt=face_rt,
        )

        # Determine the gaze state based on the EAR threshold
        for eye in ['left', 'right']:
            ear_value = compute_ear(facial_landmarks, eye)
            if ear_value < self.ear_threshold:
                gaze_vectors['eyes']['is_closed'][eye] = True
        
        # If screen's dimensions and relation to the camera are known, compute the PoG
        if (self.screen_R is not None 
            and self.screen_t is not None 
            and self.screen_height_mm is not None 
            and self.screen_width_mm is not None):

            face_pog, eyes_pog = compute_pog(
                gaze_origins,
                gaze_vectors,
                self.screen_R,
                self.screen_t,
                self.screen_width_mm,
                self.screen_height_mm,
                self.screen_width_px,
                self.screen_height_px
            )
        else:
            face_pog, eyes_pog = None, {'left': None, 'right': None}

        toc = time.perf_counter()

        # Return the result
        return GazeResult(
            # Inputs
            facial_landmarks=facial_landmarks,
            face_rt=face_rt,
            face_blendshapes=face_blendshapes,

            # Face Reconstruction
            metric_face=metric_face,
            metric_transform=metric_transform,

            # Face Gaze
            face_origin=gaze_origins['face_origin_3d'],
            face_origin_2d=gaze_origins['face_origin_2d'],
            face_gaze=gaze_vectors['face'],

            # Eye Gaze
            left=EyeResult(
                is_closed=gaze_vectors['eyes']['is_closed']['left'],
                origin=gaze_origins['eye_origins_3d']['left'],
                origin_2d=gaze_origins['eye_origins_2d']['left'],
                direction=gaze_vectors['eyes']['vector']['left'],
                pog=eyes_pog['left'],
                meta_data={
                    **gaze_vectors['eyes']['meta_data']['left']
                }
            ),
            right=EyeResult(
                is_closed=gaze_vectors['eyes']['is_closed']['right'],
                origin=gaze_origins['eye_origins_3d']['right'],
                origin_2d=gaze_origins['eye_origins_2d']['right'],
                direction=gaze_vectors['eyes']['vector']['right'],
                pog=eyes_pog['right'],
                meta_data={
                    **gaze_vectors['eyes']['meta_data']['right']
                }
            ),

            # PoG information
            pog=face_pog,

            # Meta data
            duration=toc - tic,
            eyeball_radius=self.eyeball_radius,
            eyeball_centers=self.eyeball_centers,
            perspective_matrix=self.perspective_matrix
        )
 
    def process_frame(
            self, 
            frame: np.ndarray,
        ) -> Tuple[Optional[GazeResult], Any]:

        # Start a timer
        tic = time.perf_counter()

        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = self.face_landmarker.detect(mp_image)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            return None, detection_results
        
        # Extract information fro the results
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
        face_rt = detection_results.facial_transformation_matrixes[0]
        face_blendshapes = np.array([bs.score for bs in detection_results.face_blendshapes[0]])
        
        # Perform step
        return self.step(
            face_landmarks,
            face_rt,
            face_blendshapes,
        ), detection_results