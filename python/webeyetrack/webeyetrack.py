import time
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
from .model_based import (
    obtain_eyepatch, 
    get_head_vector,
    estimate_face_width,
    estimate_camera_intrinsics,
    create_perspective_matrix,
    face_reconstruction,
    estimate_gaze_origins
)
from .data_protocols import GazeResult, TrackingStatus

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

        # Keep track of the face width cm estimate
        self.face_width_cm = None
        self.intrinsics = None

    def compute_face_origin_3d(self, image_np: np.ndarray, face_landmarks_all: np.ndarray, face_landmarks_rt: np.ndarray):

        # Get image shape
        height, width, _ = image_np.shape

        # Estimate the face width
        if self.face_width_cm is None:
            self.face_width_cm = estimate_face_width(face_landmarks_all[:, :2], face_landmarks_rt)
        if self.intrinsics is None:
            # Use the first frame to estimate the intrinsics
            self.intrinsics = estimate_camera_intrinsics(np.zeros((height, width, 3)))

        facial_landmarks_px = face_landmarks_all[:, :2] * np.array([width, height])

        perspective_matrix = create_perspective_matrix(aspect_ratio=image_np.shape[1] / image_np.shape[0])
        inv_perspective_matrix = np.linalg.inv(perspective_matrix)

        # Perform 3D face reconstruction and determine the pose in 3D centimeters
        metric_transform, metric_face = face_reconstruction(
            perspective_matrix=perspective_matrix,
            face_landmarks=face_landmarks_all[:, :3],
            face_width_cm=self.face_width_cm,
            face_rt=face_landmarks_rt,
            K=self.intrinsics,
            frame_height=height,
            frame_width=width,
            frame=image_np
        )

        # Obtain the gaze origins based on the metric face pts
        gaze_origins = estimate_gaze_origins(
            face_landmarks_3d=metric_face,
            face_landmarks=facial_landmarks_px,
        )

        return gaze_origins['face_origin_3d']
    
    def detect_facial_landmarks(self, frame: np.ndarray) -> Tuple[bool, Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]]:
        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = self.face_landmarker.detect(mp_image)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            return False, (None, None, detection_results)
        
        # Extract information fro the results
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
        face_rt = detection_results.facial_transformation_matrixes[0]
        return True, (face_landmarks, face_rt, detection_results)

    def prepare_input(self, image: np.ndarray, facial_landmarks: np.ndarray, facial_rt: np.ndarray):
        
        # Perform preprocessing to obtain the eye patch
        face_landmarks_2d = facial_landmarks[:, :2]
        face_landmarks_2d = face_landmarks_2d * np.array([image.shape[1], image.shape[0]])
        eye_patch = obtain_eyepatch(
            image, 
            face_landmarks_2d,
        )
        face_origin_3d = self.compute_face_origin_3d(
            image,
            facial_landmarks,
            facial_rt
        )

        return [
            eye_patch,
            get_head_vector(facial_rt),
            face_origin_3d
        ]
    
    def adapt(self, 
            eye_patches: list, 
            head_vectors: list,
            face_origin_3ds: list,
            pog_norms: list,
            steps_inner=5, 
            lr_inner=1e-3,
        ):
        """
        Performs MAML-style adaptation on the gaze head using support samples.
        
        Args:
            eye_patches (list): List of eye patches.
            head_vectors (list): List of head vectors.
            face_origin_3ds (list): List of face origin 3D vectors.
            steps_inner (int): Number of inner-loop adaptation steps.
            lr_inner (float): Inner-loop learning rate.
        """
        # Create the input from the samples
        support_x_eyes = np.stack(eye_patches)
        support_x_eyes = tf.convert_to_tensor(support_x_eyes, dtype=tf.float32)
        support_x_head = np.stack(head_vectors)
        support_x_head = tf.convert_to_tensor(support_x_head, dtype=tf.float32)
        support_x_face = np.stack(face_origin_3ds)
        support_x_face = tf.convert_to_tensor(support_x_face, dtype=tf.float32)
        support_x_list = [
            support_x_eyes,
            support_x_head,
            support_x_face
        ]
        support_y = np.stack(pog_norms)
        support_y = tf.convert_to_tensor(support_y, dtype=tf.float32)

        # Inner-loop gradient updates
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        losses = []
        for _ in range(steps_inner):
            with tf.GradientTape() as tape:
                preds = self.blazegaze(support_x_list, training=True)
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
    
    def adapt_from_frames(self, frames: list, norm_pogs: list, steps_inner=5, lr_inner=1e-3):
        
        # For each frame, obtain the eye patch and head vector
        eye_patches = []
        head_vectors = []
        face_origin_3ds = []
        valid_norm_pogs = []
        for frame in frames:
            status, (face_landmarks, face_rt, detection_results) = self.detect_facial_landmarks(frame)
            if not status:
                print("Failed to detect facial landmarks")
                continue

            # Perform preprocessing to obtain the eye patch
            data = self.prepare_input(
                frame,
                face_landmarks,
                face_rt
            )
            eye_patches.append(data[0])
            head_vectors.append(data[1])
            face_origin_3ds.append(data[2])
            valid_norm_pogs.append(norm_pogs)

        # Perform the adaptation
        self.adapt(
            eye_patches=eye_patches,
            head_vectors=head_vectors,
            face_origin_3ds=face_origin_3ds,
            pog_norms=valid_norm_pogs,
            steps_inner=steps_inner,
            lr_inner=lr_inner
        )

    def adapt_from_samples(self, samples, steps_inner=5, lr_inner=1e-3):
        """
        Performs MAML-style adaptation on the gaze head using support samples.
        
        Args:
            samples (dict): Dictionary with support input features and labels.
            steps_inner (int): Number of inner-loop adaptation steps.
            lr_inner (float): Inner-loop learning rate.
        """
        eye_patches = []
        head_vectors = []
        face_origin_3ds = []
        valid_norm_pogs = []
        for sample in samples:
            data = self.prepare_input(
                sample['image'],
                sample['facial_landmarks'],
                sample['facial_rt']
            )
            # Store
            eye_patches.append(data[0])
            head_vectors.append(data[1])
            face_origin_3ds.append(data[2])
            valid_norm_pogs.append(sample['pog_norm'])

        # Perform the adaptation
        self.adapt(
            eye_patches=eye_patches,
            head_vectors=head_vectors,
            face_origin_3ds=face_origin_3ds,
            pog_norms=valid_norm_pogs,
            steps_inner=steps_inner,
            lr_inner=lr_inner
        )

    def step(
            self,
            frame: np.ndarray,
            face_landmarks: np.ndarray,
            face_rt: np.ndarray
        ):
        tic = time.perf_counter()

        # Extract the 2D coordinates of the face landmarks
        face_landmarks_2d = face_landmarks[:, :2]
        face_landmarks_2d = face_landmarks_2d * np.array([frame.shape[1], frame.shape[0]])
        
        # Perform preprocessing to obtain the eye patch
        try:
            eye_patch, head_vector, face_origin_3d = self.prepare_input(
                frame,
                face_landmarks,
                face_rt
            )
        except Exception as e:
            print(f"Error in obtaining eye patch: {e}")
            return TrackingStatus.FAILED, None

        cv2.imshow('Eye Patch', eye_patch)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # Perform the gaze estimation via BlazeGaze Model
        pog_estimation = self.blazegaze.predict({
            "image": np.expand_dims(eye_patch, axis=0),
            "head_vector": np.expand_dims(head_vector, axis=0),
            "face_origin_3d": np.expand_dims(face_origin_3d, axis=0)
        }, verbose=0)
        toc = time.perf_counter()


        # return True, pog_estimation[0]
        return TrackingStatus.SUCCESS, GazeResult(
            facial_landmarks=face_landmarks,
            face_rt=face_rt,
            face_blendshapes=None,
            metric_face=None,
            metric_transform=None,
            gaze_state='open',
            pog=pog_estimation[0],
            duration=toc-tic
        )

    def process_frame(
            self,
            frame: np.ndarray
    ) -> Tuple[TrackingStatus, Optional[GazeResult], Any]:
        tic = time.perf_counter()

        # Detect the facial landmarks
        tracking_status, (face_landmarks, face_rt, detection_results) = self.detect_facial_landmarks(frame)
        if not tracking_status:
            return TrackingStatus.FAILED, None, detection_results
        
        # Perform step
        tracking_status, gaze_result = self.step(
            frame,
            face_landmarks,
            face_rt
        )
        toc = time.perf_counter()
        if tracking_status == TrackingStatus.SUCCESS:
            gaze_result.duration = toc - tic
        
        return tracking_status, gaze_result, detection_results