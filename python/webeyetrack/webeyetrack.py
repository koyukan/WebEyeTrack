import pathlib
from typing import Union, Any, Tuple, Optional
from dataclasses import dataclass

import tensorflow as tf
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

from .constants import GIT_ROOT
from .model_based import obtain_eyepatch
from .data_protocols import GazeResult, EyeResult

@dataclass
class WebEyeTrackConfig():
    blazegaze_model_fp: Union[str, pathlib.Path] = GIT_ROOT / 'python' / 'weights' / 'blazegaze_model.h5'
    ear_threshold: float = 0.2
    mediapipe_flm_model_fp: Union[str, pathlib.Path] = GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'

class WebEyeTrack():

    config: WebEyeTrackConfig

    def __init__(self, config: Optional[WebEyeTrackConfig] = None):
        if config is None:
            config = WebEyeTrackConfig()
        self.config = config

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=config.mediapipe_flm_model_fp)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Load the BlazeGaze model
        self.blazegaze = tf.keras.models.load_model(config.blazegaze_model_fp)        

    def step(
            self,
            frame: np.ndarray,
            face_landmarks: np.ndarray,
            face_rt: np.ndarray
        ):

        # Extract the 2D coordinates of the face landmarks
        face_landmarks_2d = face_landmarks[:, :2]
        face_landmarks_2d = face_landmarks_2d * np.array([frame.shape[1], frame.shape[0]])
        
        # Perform preprocessing to obtain the eye patch
        eye_patch = obtain_eyepatch(
            frame, 
            face_landmarks_2d,
        )

        cv2.imshow('Eye Patch', eye_patch)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # Perform the gaze estimation via BlazeGaze Model
        pog_estimation = self.blazegaze.predict({
            "image": np.expand_dims(eye_patch, axis=0)
        }, verbose=0)
        return pog_estimation[0]

    def process_frame(
            self,
            frame: np.ndarray
    ) -> Tuple[Optional[GazeResult], Any]:
        
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
        
        # Perform step
        # return self.step(
        #     frame,
        #     face_landmarks,
        #     face_rt
        # ), detection_results
        return None, detection_results