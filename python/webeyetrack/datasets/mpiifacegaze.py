import pathlib
from dataclasses import asdict
from typing import List, Dict, Union, Tuple, Optional
import copy
import os
import json
import pickle
from collections import defaultdict

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

class MPIIFaceGazeDataset(Dataset):
    
    def __init__(
            self, 
            dataset_dir: Union[pathlib.Path, str], 
            participants: List[int],
            face_size: Tuple[int, int] = None,
            img_size: Tuple[int, int] = None,
            dataset_size: Optional[int] = None,
            per_participant_size: Optional[int] = None
        ):

        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist."
        self.dataset_dir = dataset_dir
        
        self.img_size = img_size
        self.face_size = face_size
        self.dataset_size = dataset_size
        self.per_participant_size = per_participant_size
        self.participants = participants

        if not self.participants:
            raise ValueError("No participants were selected.")

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Determine the number of samples in the dataset
        # participant_dirs = [p for p in self.dataset_dir.iterdir() if p.is_dir()]
        participant_dirs = [self.dataset_dir / f'p{p:02d}' for p in self.participants]

        # Saving information
        self.samples = defaultdict(list)
        self.participant_calibration_data: Dict[CalibrationData] = {}

        for participant_dir in tqdm(participant_dirs, total=len(participant_dirs)):

            if self.dataset_size is not None and len(self.samples['id']) >= self.dataset_size:
                break

            participant_id = participant_dir.name
            txt_file_fp = participant_dir / f'{participant_id}.txt'
            assert txt_file_fp.is_file(), f"Participant {participant_id} does not have a txt file."

            # Load the calibration data
            cal_dir = participant_dir / 'Calibration'
            new_cal_file = cal_dir / 'calibration.pkl'
            if new_cal_file.is_file():
                with open(new_cal_file, 'rb') as f:
                    calibration_data = pickle.load(f)
            else:
                camera_mat = scipy.io.loadmat(cal_dir / 'Camera.mat')
                monitor_pose_mat = scipy.io.loadmat(cal_dir / 'monitorPose.mat')
                screen_size_mat = scipy.io.loadmat(cal_dir / 'screenSize.mat')

                # Save the calibration information
                calibration_data = CalibrationData(
                    camera_matrix=camera_mat['cameraMatrix'],
                    dist_coeffs=camera_mat['distCoeffs'],
                    camera_retval=camera_mat['retval'],
                    camera_rvecs=camera_mat['rvecs'],
                    camera_tvecs=camera_mat['tvecs'],
                    monitor_rvecs=monitor_pose_mat['rvects'],
                    monitor_tvecs=monitor_pose_mat['tvecs'],
                    monitor_height_mm=screen_size_mat['height_mm'],
                    monitor_height_px=screen_size_mat['height_pixel'],
                    monitor_width_mm=screen_size_mat['width_mm'],
                    monitor_width_px=screen_size_mat['width_pixel']
                )

                # Save the calibration data
                with open(new_cal_file, 'wb') as f:
                    pickle.dump(calibration_data, f)

            self.participant_calibration_data[participant_id] = calibration_data
            per_participant_samples = 0

            # Load the meta data
            annotations_dir = participant_dir / 'annotations'
            os.makedirs(annotations_dir, exist_ok=True)
            with open(txt_file_fp, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, total=len(lines), desc="Loading annotations"):

                    if self.dataset_size is not None and len(self.samples['id']) >= self.dataset_size:
                        break

                    if self.per_participant_size is not None and per_participant_samples >= self.per_participant_size:
                        break

                    items = line.split(' ')
                    data_complete_id = items[0].split(' ')[0].replace('.jpg', '')
                    day_id, img_id = data_complete_id.split('/')
                    data_id = f"{day_id}_{img_id}"

                    # If the annotation exists, store the path instead
                    if (annotations_dir / f"{data_id}.pkl").is_file():
                        self.samples['id'].append(data_id)
                        self.samples['participant_id'].append(participant_id)
                        self.samples['image_fp'].append(participant_dir / day_id / f"{img_id}.jpg")
                        self.samples['annotation_fp'].append(annotations_dir / f"{data_id}.pkl")
                        per_participant_samples += 1
                        continue

                    face_origin_3d = np.array(items[21:24], dtype=np.float32)
                    gaze_target_3d = np.array(items[24:27], dtype=np.float32)
                    
                    # Additionall meta data that needs to be computed
                    # Compute the 2D face origin by projecting the 3D face origin to the image plane
                    face_origin_2d, _ = cv2.projectPoints(
                        face_origin_3d, 
                        np.array([0, 0, 0], dtype=np.float32),
                        np.array([0, 0, 0], dtype=np.float32),
                        calibration_data.camera_matrix, 
                        calibration_data.dist_coeffs
                    )
                    gaze_direction_3d = gaze_target_3d - face_origin_3d

                    # Make the gaze direction a unit vector
                    gaze_direction_3d = gaze_direction_3d / np.linalg.norm(gaze_direction_3d)

                    # Create gaze_target_2d via the direction and a fixed distance
                    gaze_target_3d_semi = face_origin_3d + gaze_direction_3d * 100
                    gaze_target_2d, _ = cv2.projectPoints(
                        gaze_target_3d_semi, 
                        np.array([0, 0, 0], dtype=np.float32),
                        np.array([0, 0, 0], dtype=np.float32),
                        calibration_data.camera_matrix, 
                        calibration_data.dist_coeffs
                    )

                    # Extract the 2D facial landmarks 
                    # facial_landmarks_2d = np.array(items[3:15], dtype=np.float32).reshape(2, 6)

                    # Compute the bounding box of the face based on the facial landmarks (4 eye corners, 2 mouth corners)
                    # The bounding box is defined as the top left and bottom right corners
                    # face_bbox = np.array([
                    #     int(np.min(facial_landmarks_2d[1])), 
                    #     int(np.min(facial_landmarks_2d[0])), 
                    #     int(np.max(facial_landmarks_2d[1])), 
                    #     int(np.max(facial_landmarks_2d[0]))
                    # ])

                    # Loading the image
                    image_fp = participant_dir / items[0]
                    image = Image.open(image_fp)
                    image_np = cv2.imread(str(image_fp))

                    # Look at the image
                    # cv2.imshow('image', image_np)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # If the face landmark already exists, load it instead of computing it
                    face_landmarks_dir = participant_dir / 'face_landmarks'
                    os.makedirs(face_landmarks_dir, exist_ok=True)
                    detection_fp = face_landmarks_dir / f"{data_id}_detection.pkl"
                    face_landmarks_rt_fp = face_landmarks_dir / f"{data_id}_rt.npy"
                    face_landmarks_fp = face_landmarks_dir / f"{data_id}.npy"
                    face_blendshapes_fp = face_landmarks_dir / f"{data_id}_blendshapes.npy"

                    if face_landmarks_rt_fp.is_file() and face_landmarks_fp.is_file() and face_blendshapes_fp.is_file() and detection_fp.is_file():
                        face_landmarks_rt = np.load(face_landmarks_rt_fp)
                        face_landmarks_all = np.load(face_landmarks_fp)
                        face_landmarks = np.array([[lm[0] * image.size[0], lm[1] * image.size[1]] for lm in face_landmarks_all])
                        face_blendshapes = np.load(face_blendshapes_fp, allow_pickle=True)
                        with open(detection_fp, 'rb') as f:
                            detection_results = pickle.load(f)
                    else:
                        
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
                        with open(face_landmarks_rt_fp, 'wb') as f:
                            np.save(f, face_landmarks_rt)
                        with open(face_landmarks_fp, 'wb') as f:
                            np.save(f, face_landmarks_all)
                        with open(face_blendshapes_fp, 'wb') as f:
                            np.save(f, face_blendshapes)
                        with open(detection_fp, 'wb') as f:
                            pickle.dump(detection_results, f)

                        face_landmarks = np.array([[lm.x * image.size[0], lm.y * image.size[1]] for lm in face_landmarks_proto])

                    # Compute the bounding box
                    face_bbox = np.array([
                        int(np.min(face_landmarks[:, 1])), 
                        int(np.min(face_landmarks[:, 0])), 
                        int(np.max(face_landmarks[:, 1])), 
                        int(np.max(face_landmarks[:, 0]))
                    ])

                    # Convert face_blendshapes to a proper numpy array
                    np_face_blendshapes = np.array([x.score for x in face_blendshapes])

                    # Extract the PoG
                    pog_px=np.array(items[1:3], dtype=np.float32)

                    # Compute the normalized PoG
                    pog_norm = np.array([
                        pog_px[0] / calibration_data.monitor_width_px,
                        pog_px[1] / calibration_data.monitor_height_px
                    ])

                    # Compute the PoG in mm
                    pog_mm = np.array([
                        pog_norm[0] * calibration_data.monitor_width_mm,
                        pog_norm[1] * calibration_data.monitor_height_mm
                    ]).flatten()

                    annotation = Annotations(
                        original_img_size=np.array(image_np.shape),
                        intrinsics=calibration_data.camera_matrix,
                        # Facial landmarks
                        facial_detection_results=detection_results,
                        facial_landmarks=face_landmarks_all,
                        facial_landmarks_2d=face_landmarks,
                        facial_rt=face_landmarks_rt,
                        face_blendshapes=np_face_blendshapes,
                        face_bbox=face_bbox,
                        head_pose_3d=np.array(items[15:21], dtype=np.float32).reshape(3, 2),
                        # Face Gaze
                        face_origin_3d=face_origin_3d,
                        face_origin_2d=face_origin_2d.flatten(),
                        face_gaze_vector=gaze_direction_3d,
                        # Eye Gaze
                        left_eye_origin_3d=np.empty((3, 0), dtype=np.float32),
                        left_eye_origin_2d=np.empty((2, 0), dtype=np.float32),
                        left_gaze_vector=np.empty((3, 0), dtype=np.float32),
                        right_eye_origin_3d=np.empty((3, 0), dtype=np.float32),
                        right_eye_origin_2d=np.empty((2, 0), dtype=np.float32),
                        right_gaze_vector=np.empty((3, 0), dtype=np.float32),
                        # Target information
                        gaze_target_3d=gaze_target_3d,
                        gaze_target_2d=gaze_target_2d.flatten(),
                        pog_px=pog_px,
                        pog_norm=pog_norm,
                        pog_mm=pog_mm,
                        # Gaze State Information
                        is_closed=np.array([False])
                    )
                    
                    # Save the annotation
                    with open(annotations_dir / f"{data_id}.pkl", 'wb') as f:
                        pickle.dump(annotation, f)

                    self.samples['id'].append(data_id)
                    self.samples['participant_id'].append(participant_id)
                    self.samples['image_fp'].append(pathlib.Path(image_fp))
                    self.samples['annotation_fp'].append(annotations_dir / f"{data_id}.pkl")
                    per_participant_samples += 1
                    
        # Convert the samples to a DataFrame
        self.samples = pd.DataFrame(self.samples)

    def get_samples_meta_df(self):
        return self.samples
            
    def __getitem__(self, index: int):
        # Make a copy of the sample
        # sample = copy.deepcopy(self.samples[index])
        sample = self.samples.iloc[index]

        # Create torch-compatible data
        image = Image.open(sample.image_fp)
        image_np = np.array(image)

        # Get the calibration
        calibration_data = self.participant_calibration_data[sample.participant_id]

        # Load the annotations
        with open(sample.annotation_fp, 'rb') as f:
            annotations = pickle.load(f)
 
        # Convert from uint8 to float32
        image_np = image_np.astype(np.float32) / 255.0

        # Draw the facial landmarks on the image
        # annotated_img = draw_landmarks_on_image(image_np, detection_results)
        # cv2.imshow('annotated_img', annotated_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Crop out the face image and resize to have standard size
        face_bbox = annotations.face_bbox
        
        # Clip the face bounding box to the image size and avoid negative indexing
        face_bbox[0] = np.clip(face_bbox[0], 0, image.size[1] - 1)
        face_bbox[1] = np.clip(face_bbox[1], 0, image.size[0] - 1)
        face_bbox[2] = np.clip(face_bbox[2], 0, image.size[1] - 1)
        face_bbox[3] = np.clip(face_bbox[3], 0, image.size[0] - 1)
        face_image_np = image_np[face_bbox[0]:face_bbox[2], face_bbox[1]:face_bbox[3]]
        if self.face_size is not None:
            face_image_np = cv2.resize(face_image_np, self.face_size, interpolation=cv2.INTER_LINEAR)

        # Load the texture
        # texture_dir = self.dataset_dir / sample.participant_id / 'textures'
        # texture_fp = texture_dir / f"{data_id}.jpg"
        # texture = cv2.imread(str(texture_fp))
        # texture = texture.astype(np.float32) / 255.0

        # Compute the relative gaze direction based on the facial landmarks
        facelandmark_rt = np.load(self.dataset_dir / sample.participant_id / 'face_landmarks' / f"{sample.id}_rt.npy")
        head_direction_rotation = facelandmark_rt[0:3, 0:3]
        head_direction_xyz = Rotation.from_matrix(head_direction_rotation).as_rotvec()
        head_direction_xyz = head_direction_xyz / np.linalg.norm(head_direction_xyz)
        relative_gaze_vector = annotations.face_gaze_vector - head_direction_xyz
        relative_gaze_vector = relative_gaze_vector / np.linalg.norm(relative_gaze_vector)

        # Visualize the face image
        # cv2.imshow('face_image', face_image_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Resize the raw input image if needed
        if self.img_size is not None:
            image_np = cv2.resize(image_np, self.img_size, interpolation=cv2.INTER_LINEAR)
            annotations = resize_annotations(annotations, image.size, self.img_size)
            intrinsics = resize_intrinsics(calibration_data.camera_matrix, image.size, self.img_size)
        else:
            intrinsics = calibration_data.camera_matrix
        
        # Revert the image to the correct format
        image_np = np.moveaxis(image_np, -1, 0)
        face_image_np = np.moveaxis(face_image_np, -1, 0)
        # texture = np.moveaxis(texture, -1, 0)

        item_dict = {
            'participant_id': sample.participant_id,
            'image': image_np,
            'face_image': face_image_np,
            # 'uv_texture': texture,
            'intrinsics': intrinsics,
            'dist_coeffs': calibration_data.dist_coeffs,
            'screen_R': calibration_data.monitor_rvecs.astype(np.float32),
            'screen_t': calibration_data.monitor_tvecs.astype(np.float32),
            'screen_height_mm': calibration_data.monitor_height_mm.astype(np.float32),
            'screen_height_px': calibration_data.monitor_height_px.astype(np.float32),
            'screen_width_mm': calibration_data.monitor_width_mm.astype(np.float32),
            'screen_width_px': calibration_data.monitor_width_px.astype(np.float32),
            # 'gaze_origin_depth_mean': self.gaze_origin_depth_mean,
            # 'gaze_origin_depth_std': self.gaze_origin_depth_std,
            # 'pog_px_mean': self.pog_px_mean,
            # 'pog_px_std': self.pog_px_std,
            # 'mediapipe_head_vector': head_direction_xyz.astype(np.float32),
            # 'relative_gaze_vector': relative_gaze_vector.astype(np.float32)
        }
        item_dict.update(asdict(annotations))
        return item_dict

    def __len__(self):
        return len(self.samples)

    # def to_df(self):

    #     data = defaultdict(list)
    #     for i in range(len(self.samples)):
    #         sample = self[i]
    #         for k, v in sample.items():
    #             data[k].append(v)

    #     return pd.DataFrame(data)

if __name__ == '__main__':

    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        dataset_size=2,
        participants=[1]
    )
    print(len(dataset))

    sample = dataset[0]
    # print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))
    # print(sample.keys())
    print(json.dumps({k: str(v) for k, v in sample.items()}, indent=4))
