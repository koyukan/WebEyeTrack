"""
References
https://github.com/swook/faze_preprocess/blob/master/create_hdf_files_for_faze.py
"""

import pathlib
from typing import List, Dict, Union, Optional, Tuple
import shutil
import json
from dataclasses import asdict
import pickle

import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import yaml
import numpy as np
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from webeyetrack.constants import GIT_ROOT
from webeyetrack.data_protocols import Annotations, CalibrationData, Sample

CWD = pathlib.Path(__file__).parent
LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]

class GazeCaptureDataset(Dataset):

    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        face_size: Tuple[int, int] = None,
        img_size: Tuple[int, int] = None,
        dataset_size: Optional[int] = None,
        per_participant_size: Optional[int] = None,
    ):
        
        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist."
        self.dataset_dir = dataset_dir
        self.dataset_size = dataset_size
        self.per_participant_size = per_participant_size
        self.face_size = face_size
        self.img_size = img_size

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
        num_samples = 0
        for part_dir in tqdm(participant_dir, total=len(participant_dir)):

            if self.dataset_size is not None and num_samples >= self.dataset_size:
                break

            participant_id = part_dir.name
            frames_json = part_dir / 'frames.json'
            with open(frames_json, 'r') as f:
                frames_fname_list = json.load(f)
            with open(part_dir / 'dotInfo.json', 'r') as f:
                dot_info = json.load(f)
            dot_info_df = pd.DataFrame(dot_info)
            
            per_participant_samples = 0

            for i, frame in tqdm(enumerate(frames_fname_list), total=len(frames_fname_list)):

                if self.dataset_size is not None and num_samples >= self.dataset_size:
                    break

                if self.per_participant_size is not None and per_participant_samples >= self.per_participant_size:
                    break

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

                # Convert face_blendshapes to a proper numpy array
                np_face_blendshapes = np.array([x.score for x in face_blendshapes])

                # Create the annotation
                annotation = Annotations(
                    original_img_size=np.array(image_np.shape),
                    facial_landmarks=face_landmarks_all,
                    facial_landmarks_2d=face_landmarks,
                    facial_rt=face_landmarks_rt,
                    face_blendshapes=np_face_blendshapes,
                    face_bbox=face_bbox,
                    head_pose_3d=np.array([0, 0, 0, 0, 0, 0]), # From MPIIFaceGaze, not GazeCapture
                    face_origin_3d=face_origin_3d,
                    face_origin_2d=face_origin_2d,
                    gaze_direction_3d=face_gaze_vector,
                    gaze_target_3d=gaze_3d_target,
                    gaze_target_2d=np.array([dot_info['XCam'], dot_info['YCam']]),
                    pog_px=np.array([dot_info['XPts'], dot_info['YPts']])
                )

                # Create a sample
                sample = Sample(
                    participant_id=participant_id,
                    image_fp=frame_fp,
                    annotations=annotation,
                )

                self.samples.append(sample)
                num_samples += 1
                per_participant_samples += 1

    def __getitem__(self, index: int):
        sample = self.samples.iloc[index]

        # Load image
        image = Image.open(sample.image_fp)
        image_np = np.array(image)

        # Get the calibration
        calibration_data = self.participant_calibration_data[sample.participant_id]

        # Load the annotations
        with open(sample.annotation_fp, 'rb') as f:
            annotations = pickle.load(f)

        item_dict = {
            'person_id': sample.participant_id,
            'image': image_np,
            'intrinsics': calibration_data.camera_matrix,
            'dist_coeffs': calibration_data.dist_coeffs,
            'screen_RT': calibration_data.screen_RT.astype(np.float32),
            'screen_height_cm': calibration_data.monitor_height_cm,
            'screen_height_px': calibration_data.monitor_height_px,
            'screen_width_cm': calibration_data.monitor_width_cm,
            'screen_width_px': calibration_data.monitor_width_px,
        }
        item_dict.update(asdict(annotations))
        return item_dict

    def __len__(self):
        return len(self.samples)
    
# if __name__ == "__main__":
    
#     from ..constants import DEFAULT_CONFIG
#     with open(DEFAULT_CONFIG, 'r') as f:
#         config = yaml.safe_load(f)

#     dataset = GazeCaptureDataset(
#         dataset_dir=GIT_ROOT / pathlib.Path(config['datasets']['GazeCapture']['path']),
#         per_participant_size=10,
#         dataset_size=20
#     )
#     print(len(dataset))

#     sample = dataset[0]
#     print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))