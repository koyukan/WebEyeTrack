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

        # Freeze encoder layers
        for layer in self.blazegaze.layers:
            if 'cnn_encoder' in layer.name or 'encoder' in layer.name:
                layer.trainable = False

    def adapt(self, samples, steps_inner=5, lr_inner=1e-3):
        """
        Performs MAML-style adaptation on the gaze head using support samples.
        
        Args:
            samples (dict): Dictionary with support input features and labels.
            steps_inner (int): Number of inner-loop adaptation steps.
            lr_inner (float): Inner-loop learning rate.
        """
        # Perform the eye patch extraction
        eye_patches = []
        for sample in samples:
            # Perform preprocessing to obtain the eye patch
            frame = sample['image']
            face_landmarks_2d = sample['facial_landmarks'][:, :2]
            face_landmarks_2d = face_landmarks_2d * np.array([frame.shape[1], frame.shape[0]])
            eye_patch = obtain_eyepatch(
                frame, 
                face_landmarks_2d,
            )
            eye_patches.append(eye_patch)

        # Create the input from the samples
        support_x = np.stack(eye_patches)
        support_x = tf.convert_to_tensor(support_x, dtype=tf.float32)
        support_y = np.stack([y['pog_cm'] for y in samples])
        support_y = tf.convert_to_tensor(support_y, dtype=tf.float32)

        # Inner-loop gradient updates
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        losses = []
        for _ in range(steps_inner):
            with tf.GradientTape() as tape:
                preds = self.blazegaze([support_x], training=True)
                loss = loss_fn(support_y, preds)
            grads = tape.gradient(loss, self.blazegaze.trainable_variables)
            losses.append(loss)

            # Filter grads to only update gaze head
            # import pdb; pdb.set_trace()
            trainable_vars = [v for v in self.blazegaze.trainable_variables]
            gaze_grads = [g for g, v in zip(grads, self.blazegaze.trainable_variables)]

            for var, grad in zip(trainable_vars, gaze_grads):
                if grad is not None:
                    var.assign_sub(lr_inner * grad)

        # Report the loss
        print(f"Adaptation Loss: {losses}")

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
        try:
            eye_patch = obtain_eyepatch(
                frame, 
                face_landmarks_2d,
            )
        except Exception as e:
            print(f"Error in obtaining eye patch: {e}")
            return False, None

        cv2.imshow('Eye Patch', eye_patch)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # Perform the gaze estimation via BlazeGaze Model
        pog_estimation = self.blazegaze.predict({
            "image": np.expand_dims(eye_patch, axis=0)
        }, verbose=0)
        return True, pog_estimation[0]

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