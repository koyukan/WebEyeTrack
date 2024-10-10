"""
References
https://github.com/swook/faze_preprocess/blob/master/create_hdf_files_for_faze.py
"""

import pathlib
from typing import List, Dict, Union, Optional
import shutil
import json

import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2
from PIL import Image
import scipy.io
import yaml
import numpy as np
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..constants import GIT_ROOT
from ..vis import draw_gaze_origin
from ..data_protocols import Annotations, CalibrationData, Sample
from .utils import resize_annotations, resize_intrinsics, draw_landmarks_on_image, compute_uv_texture

CWD = pathlib.Path(__file__).parent
LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]

class GazeCaptureDataset(Dataset):

    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        dataset_size: Optional[int] = None,
    ):
        
        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        self.dataset_dir = dataset_dir
        assert self.dataset_dir.is_dir(), f"Dataset directory {self.dataset_dir} does not exist."
        self.dataset_size = dataset_size

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.preprocessing()

    def preprocessing(self):

        gz_elements = [x for x in self.dataset_dir.iterdir() if x.suffix == '.gz']
        for gz_element in tqdm(gz_elements, total=len(gz_elements)):
            # If element is a .tar.gz file, extract it
            if gz_element.suffix == '.gz':
                shutil.unpack_archive(gz_element, extract_dir=self.dataset_dir)
                gz_element.unlink()

        # Now obtain each participants folder
        participant_dir = [x for x in self.dataset_dir.iterdir() if x.is_dir()]

        # Saving information
        self.samples: List[Sample] = []
        self.participant_calibration_data: Dict[CalibrationData] = {}

        # Open the ``frames.json`` file with a list of frame names
        for part_dir in participant_dir:
            participant_id = part_dir.name
            frames_json = part_dir / 'frames.json'
            with open(frames_json, 'r') as f:
                frames_fname_list = json.load(f)
            with open(part_dir / 'dotInfo.json', 'r') as f:
                dot_info = json.load(f)
            dot_info_df = pd.DataFrame(dot_info)

            for i, frame in enumerate(frames_fname_list):
                frame_fp = part_dir / 'frames' / frame
                dot_info = dot_info_df.iloc[i]

                # Load the image
                image_np = cv2.imread(str(frame_fp))

                # Detect the facial landmarks via MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
                detection_results = self.face_landmarker.detect(mp_image)

                # Compute the face bounding box based on the MediaPipe landmarks
                try:
                    face_landmarks_proto = detection_results.face_landmarks[0]
                except:
                    # print(f"Participant {participant_id} image {items[0]} does not have a face detected.")
                    continue

                # Save the detection results as numpy arrays
                face_landmarks_all = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
                face_landmarks_rt = detection_results.facial_transformation_matrixes[0]
                face_blendshapes = detection_results.face_blendshapes[0]
                face_landmarks = np.array([[lm.x * image_np.shape[0], lm.y * image_np.shape[1]] for lm in face_landmarks_proto])
                
                # Draw the landmarks on the image
                # image_landmarks = draw_landmarks_on_image(image_np, detection_results)
                # cv2.imshow('image', image_landmarks)
                # cv2.waitKey(0)

                # Compute the gaze origin (between the eyes) in 2D
                data_2d_pairs = {
                    'left': face_landmarks[LEFT_EYE_LANDMARKS] * np.array([image_np.shape[0], image_np.shape[1]]),
                    'right': face_landmarks[RIGHT_EYE_LANDMARKS] * np.array([image_np.shape[0], image_np.shape[1]])
                }

                # Compute the eye origin in 2D
                eye_origins_2d = {}
                for k, v in data_2d_pairs.items():
                    eye_origins_2d[k] = np.mean(v, axis=0)

                # Compute the face origin in 2D
                face_origin_2d = np.mean([eye_origins_2d['left'], eye_origins_2d['right']], axis=0)

                # Compute the bounding box
                face_bbox = np.array([
                    int(np.min(face_landmarks[:, 1])), 
                    int(np.min(face_landmarks[:, 0])), 
                    int(np.max(face_landmarks[:, 1])), 
                    int(np.max(face_landmarks[:, 0]))
                ])

                # Apply the facial translation to the landmarks
                t_face_landmarks_xyz = face_landmarks_all[:, :3].copy()
                t_face_landmarks_xyz = t_face_landmarks_xyz + face_landmarks_rt[:3, 3]

                # Compute the gaze origin (between the eyes)
                data_3d_pairs = {
                    'left': t_face_landmarks_xyz[LEFT_EYE_LANDMARKS][:, :3],
                    'right': t_face_landmarks_xyz[RIGHT_EYE_LANDMARKS][:, :3]
                }

                # Create the 3D gaze target from the dot
                gaze_3d_target = np.array([dot_info['XCam'], dot_info['YCam'], 0])
                # import pdb; pdb.set_trace()

                # Compute the 3D eye origins and vectors
                origins_3d = {}
                eye_gaze_vectors = {}
                for k, v in data_3d_pairs.items():
                    origins_3d[k] = np.mean(v, axis=0)

                    # Compute the gaze vector from the eye origin to the gaze target
                    gaze_vector = gaze_3d_target - origins_3d[k]
                    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
                    eye_gaze_vectors[k] = gaze_vector

                # Compute the face gaze vector
                face_gaze_vector = np.mean([eye_gaze_vectors['left'], eye_gaze_vectors['right']], axis=0)
                face_gaze_vector = face_gaze_vector / np.linalg.norm(face_gaze_vector)

                # Compute the face origin
                face_origin_3d = np.mean([origins_3d['left'], origins_3d['right']], axis=0)

                import pdb; pdb.set_trace()

                # Create the annotation
                annotation = Annotations(
                    original_img_size=np.array(image_np.shape),
                    facial_landmarks=face_landmarks_all,
                    facial_landmarks_2d=face_landmarks,
                    facial_rt=face_landmarks_rt,
                    face_blendshapes=face_blendshapes,
                    face_bbox=face_bbox,
                    head_pose_3d=np.array([0, 0, 0, 0, 0, 0]), # From MPIIFaceGaze, not GazeCapture
                    face_origin_3d=face_origin_3d,
                    face_origin_2d=face_origin_2d,
                    gaze_direction_3d=face_gaze_vector,
                    gaze_target_3d=gaze_3d_target,
                    gaze_target_2d=np.array([dot_info['XCam'], dot_info['YCam']]),
                )

                # Create a sample
                # sample = Sample(
                # )

                # self.samples.append(sample)

            break

    def __getitem__(self, idx):
        return {}

    def __len__(self):
        return len(self.samples)
    
if __name__ == "__main__":
    
    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = GazeCaptureDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['GazeCapture']['path']),
        dataset_size=1000,
    )
    print(len(dataset))

    sample = dataset[0]
    print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))