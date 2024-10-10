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
from .data_protocols import Annotations, CalibrationData, Sample
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
        base_options = python.BaseOptions(model_asset_path=str(CWD / 'face_landmarker_v2_with_blendshapes.task'))
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
        # self.annotations: Dict[str, Annotations] = {}
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

            # import pdb; pdb.set_trace()

            for frame in frames_fname_list:
                frame_fp = part_dir / 'frames' / frame

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

                # Compute the 3D eye origins
                origins_3d = {}
                for k, v in data_3d_pairs.items():
                    origins_3d[k] = np.mean(v, axis=0)

                # Take the average to get the gaze origin
                gaze_origin = (origins_3d['left'] + origins_3d['right']) / 2

                # Draw the gaze origin on the image

                # Draw the landmarks on the image
                # image_landmarks = draw_landmarks_on_image(image_np, detection_results)
                # cv2.imshow('image', image_landmarks)
                # cv2.waitKey(0)

                # Create the annotation
                annotation = Annotations(
                )
                # self.annotations[participant_id]

            break

    def __getitem__(self, idx):
        return {}

    def __len__(self):
        return self.dataset_size
    
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