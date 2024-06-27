import pathlib
from dataclasses import asdict
from typing import List, Dict, Union
import copy

import cv2
from PIL import Image
import scipy.io
import yaml
import numpy as np
from torch.utils.data import Dataset

from ..constants import GIT_ROOT

from .data_protocols import Annotations, CalibrationData, Sample
from .utils import resize_annotations

class MPIIFaceGazeDataset(Dataset):
    
    def __init__(self, dataset_dir: Union[pathlib.Path, str]):
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        self.dataset_dir = dataset_dir
        assert self.dataset_dir.is_dir(), f"Dataset directory {self.dataset_dir} does not exist."

        # Determine the number of samples in the dataset
        participant_dirs = [p for p in self.dataset_dir.iterdir() if p.is_dir()]

        # Saving information
        self.samples: List[Sample] = []
        self.participant_calibration_data: Dict[CalibrationData] = {}

        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            txt_file_fp = participant_dir / f'{participant_id}.txt'
            assert txt_file_fp.is_file(), f"Participant {participant_id} does not have a txt file."

            # Load the calibration data
            cal_dir = participant_dir / 'Calibration'
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
            self.participant_calibration_data[participant_id] = calibration_data

            # Load the meta data
            annotations = {} 
            with open(txt_file_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(' ')

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

                    # gaze_target_2d, _ = cv2.projectPoints(
                    #     gaze_target_3d, 
                    #     np.array([0, 0, 0], dtype=np.float32),
                    #     np.array([0, 0, 0], dtype=np.float32),
                    #     calibration_data.camera_matrix, 
                    #     calibration_data.dist_coeffs
                    # )

                    # Create gaze_target_2d via the direction and a fixed distance
                    gaze_target_3d_semi = face_origin_3d + gaze_direction_3d / 5
                    gaze_target_2d, _ = cv2.projectPoints(
                        gaze_target_3d_semi, 
                        np.array([0, 0, 0], dtype=np.float32),
                        np.array([0, 0, 0], dtype=np.float32),
                        calibration_data.camera_matrix, 
                        calibration_data.dist_coeffs
                    )
                    # import pdb; pdb.set_trace()

                    annotation = Annotations(
                        pog_px=np.array(items[1:3], dtype=np.float32),
                        facial_landmarks_2d=np.array(items[3:15], dtype=np.float32).reshape(2, 6),
                        head_pose_3d=np.array(items[15:21], dtype=np.float32).reshape(3, 2),
                        face_origin_3d=face_origin_3d,
                        face_origin_2d=face_origin_2d.flatten(),
                        gaze_target_3d=gaze_target_3d,
                        gaze_target_2d=gaze_target_2d.flatten(),
                        gaze_direction_3d=gaze_direction_3d,
                        which_eye=items[27]
                    )
            
                    annotations[items[0]] = annotation
     
            day_folders = [d for d in participant_dir.glob('day*') if d.is_dir()]
            for day_folder in day_folders:

                # Load the images
                images = [f for f in day_folder.glob('*.jpg') if f.is_file()]
                for image in images:
                    complete_name = f"{day_folder.name}/{image.name}"
                    self.samples.append(
                        Sample(
                            image_fp=image,
                            annotations=annotations[complete_name]
                        )
                    )
            
    def __getitem__(self, index: int):
        # Make a copy of the sample
        sample = copy.deepcopy(self.samples[index])

        # Create torch-compatible data
        image = Image.open(sample.image_fp)
        image_np = np.array(image)

        # Reshape image to match 640x480
        image_np = cv2.resize(image_np, (640, 480), interpolation=cv2.INTER_LINEAR)
        image_np = np.moveaxis(image_np, -1, 0)

        # Modify the annotations to account for the image resizing
        sample.annotations = resize_annotations(sample.annotations, image.size, (640, 480))

        # Convert from uint8 to float32
        image_np = image_np.astype(np.float32) / 255.0

        sample_dict = {
            'image': image_np,
        }
        sample_dict.update(asdict(sample.annotations))
        return sample_dict

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':

    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = MPIIFaceGazeDataset(GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']))
    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())