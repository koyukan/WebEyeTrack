import pathlib
from dataclasses import dataclass
from typing import List, Dict

import scipy.io
import yaml
import numpy as np
from torch.utils.data import Dataset

from ..constants import GIT_ROOT

FILE_DIR = pathlib.Path(__file__).parent

with open(FILE_DIR.parent / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

@dataclass
class Annotations:
    pog_px: np.ndarray # (2,)
    facial_landmarks: np.ndarray # (2, 6)
    head_pose: np.ndarray # (6,), rotation matrix
    face_origin: np.ndarray # (3,)
    gaze_target: np.ndarray # (3,)
    which_eye: str # (left, right)

@dataclass
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    camera_retval: float
    camera_rvecs: np.ndarray
    camera_tvecs: np.ndarray
    monitor_rvecs: np.ndarray
    monitor_tvecs: np.ndarray
    monitor_height_mm: float
    monitor_height_px: float
    monitor_width_mm: float
    monitor_width_px: float

@dataclass
class Sample:
    image_fp: pathlib.Path
    annotations: Annotations

class MPIIFaceGazeDataset(Dataset):
    
    def __init__(self, dataset_dir: pathlib.Path):
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
            self.participant_calibration_data[participant_id] = CalibrationData(
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

            # Load the meta data
            annotations = {} 
            with open(txt_file_fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(' ')
                    annotations[items[0]] = Annotations(
                        pog_px=np.array(items[1:3], dtype=np.float32),
                        facial_landmarks=np.array(items[3:15], dtype=np.float32).reshape(2, 6),
                        head_pose=np.array(items[15:21], dtype=np.float32).reshape(3, 2),
                        face_origin=np.array(items[21:24], dtype=np.float32),
                        gaze_target=np.array(items[24:27], dtype=np.float32),
                        which_eye=items[27]
                    )
         
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
        ...

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':

    dataset = MPIIFaceGazeDataset(GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']))
    print(len(dataset))