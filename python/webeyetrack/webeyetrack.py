import time
from typing import Dict, Any, Literal, Optional, Tuple
from collections import deque

import tensorflow as tf
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
from .utilities import transform_3d_to_3d

PARAMETER_LIST = [
    'frame_height',
    'frame_width',
    'face_width_cm',
    'intrinsics',
    'screen_RT',
    'screen_width_cm',
    'screen_height_cm',
    'screen_width_px',
    'screen_height_px',
]

# Custom weight initialization: Identity Mapping
def identity_initializer(shape, dtype=None):
    """ Initializes weights close to identity transformation. """
    W = np.eye(shape[0], shape[1])  # Create an identity matrix
    if W.shape != shape:
        W = np.pad(W, [(0, shape[0] - W.shape[0]), (0, shape[1] - W.shape[1])], mode='constant')  # Padding if necessary
    return tf.convert_to_tensor(W, dtype=tf.float32)

class WebEyeTrack():

    def __init__(
            self, 
            model_asset_path: str, 
            frame_height: Optional[int] = None,
            frame_width: Optional[int] = None,
            intrinsics: Optional[np.ndarray] = None,
            face_width_cm: Optional[float] = None,
            screen_RT: Optional[np.ndarray] = None,
            screen_width_cm: Optional[float] = None,
            screen_height_cm: Optional[float] = None,
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
        self.screen_RT = screen_RT
        self.screen_width_cm = screen_width_cm
        self.screen_height_cm = screen_height_cm
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px

        # State variables
        self.prior_gaze = None
        self.prior_depth = None

        # # Create MLP for gaze correction
        # self.mlp = tf.keras.Sequential([
        #     tf.keras.layers.Dense(4, kernel_initializer=identity_initializer, bias_initializer='zeros', activation='relu', input_shape=(2,)),
        #     tf.keras.layers.Dense(2, kernel_initializer=identity_initializer, bias_initializer='zeros')
        # ])
        # self.mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #             loss='mse')

    def config(self, **kwargs):
        for key, value in kwargs.items():
            if key in PARAMETER_LIST:
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid key: {key} in {PARAMETER_LIST}')

    # def calibrate(self, samples):

    #     def loss_fn(params):
    #         # The loss function is the sum of the squared differences between the esimated PoG and the ground truth PoG
    #         # For all samples, compute the error
    #         errors = []
    #         for sample in samples:

    #             # Use the current parameters to compute the PoG
    #             # eyeball_centers = (np.array([params[1], params[2], params[3]]), np.array([-params[1], params[2], params[3]]))
    #             eyeball_centers = (np.array([params[0], params[1], params[2]]), np.array([-params[0], params[1], params[2]]))
    #             # eyeball_radius = params[0]
    #             results = self.process_sample(sample['image'], sample, eyeball_centers=eyeball_centers)

    #             # Compute the error
    #             error = np.linalg.norm(results.pog_cm_s - sample['pog_cm'].reshape((2)))
    #             errors.append(error)

    #         return sum(errors)

    #     dimensions = [
    #         # Real(15/2, 25/2, name='eyeball_radius'),
    #         Real(EYEBALL_X/2, EYEBALL_X*2, name='eyeball_x'),
    #         Real(EYEBALL_Y/2, EYEBALL_Y*2, name='eyeball_y'),
    #         Real(EYEBALL_Z/2, EYEBALL_Z*2, name='eyeball_z')
    #     ]

    #     # Initial guess for the parameters
    #     # x0 = [EYEBALL_RADIUS, EYEBALL_X, EYEBALL_Y, EYEBALL_Z]
    #     x0 = [EYEBALL_X, EYEBALL_Y, EYEBALL_Z]

    #     # Perform the optimization
    #     result = gp_minimize(
    #         func=loss_fn,
    #         dimensions=dimensions,
    #         x0=x0,
    #         n_calls=11,
    #         random_state=42
    #     )

    #     print(result)
    #     import pdb; pdb.set_trace() # TODO

    def online_train(self, estimated_pog, gt_pog):
        """ Perform an online update with a single (x, y) pair. """
        x_train = np.array([estimated_pog])  # Convert single sample to NumPy array
        y_train = np.array([gt_pog])  # Ground truth

        # Train on one sample (1 step)
        self.mlp.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
        
    def step(
            self, 
            facial_landmarks, 
            face_rt, 
            face_blendshapes, 
            frame=None
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
            frame_width=self.frame_width,
            frame=frame
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
        if (self.screen_RT is not None 
            and self.screen_height_cm is not None 
            and self.screen_width_cm is not None):

            face_pog, eyes_pog = compute_pog(
                gaze_origins,
                gaze_vectors,
                self.screen_RT,
                self.screen_width_cm,
                self.screen_height_cm,
                self.screen_width_px,
                self.screen_height_px
            )

            # # Perform gaze correction
            # pog_norm = face_pog.pog_norm
            # corrected_pog_norm = self.mlp.predict(np.array([pog_norm]), verbose=0).flatten()

            # # Convert norm to pixels and cm versions of PoG
            # corrected_pog_cm_s = corrected_pog_norm * np.array([self.screen_width_cm, self.screen_height_cm])
            # corrected_pog_px = corrected_pog_norm * np.array([self.screen_width_px, self.screen_height_px])
            # corrected_pog_cm_c = transform_3d_to_3d(np.append(corrected_pog_cm_s, 0).reshape((-1,3)), self.screen_RT).flatten()

            # # Update the PoG information (for the eyes as well)
            # face_pog.pog_norm = corrected_pog_norm
            # face_pog.pog_cm_s = corrected_pog_cm_s
            # face_pog.pog_cm_c = corrected_pog_cm_c
            # face_pog.pog_px = corrected_pog_px
            # eyes_pog['left'].pog_norm = corrected_pog_norm
            # eyes_pog['left'].pog_cm_s = corrected_pog_cm_s
            # eyes_pog['left'].pog_cm_c = corrected_pog_cm_c
            # eyes_pog['left'].pog_px = corrected_pog_px
            # eyes_pog['right'].pog_norm = corrected_pog_norm
            # eyes_pog['right'].pog_cm_s = corrected_pog_cm_s
            # eyes_pog['right'].pog_cm_c = corrected_pog_cm_c
            # eyes_pog['right'].pog_px = corrected_pog_px

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
            # frame=frame
        ), detection_results