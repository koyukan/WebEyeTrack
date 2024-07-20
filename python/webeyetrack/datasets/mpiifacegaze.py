import pathlib
from dataclasses import asdict
from typing import List, Dict, Union, Tuple, Optional
import copy

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
from .utils import resize_annotations, resize_intrinsics, draw_landmarks_on_image

CWD = pathlib.Path(__file__).parent

class MPIIFaceGazeDataset(Dataset):
    
    def __init__(
            self, 
            dataset_dir: Union[pathlib.Path, str], 
            dataset_size: Optional[int] = None,
            face_size: tuple = (224, 224),
            img_size: Optional[Tuple] = None,
        ):

        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        self.dataset_dir = dataset_dir
        assert self.dataset_dir.is_dir(), f"Dataset directory {self.dataset_dir} does not exist."
        self.img_size = img_size
        self.face_size = face_size
        self.dataset_size = dataset_size

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=str(CWD / 'face_landmarker_v2_with_blendshapes.task'))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Determine the number of samples in the dataset
        participant_dirs = [p for p in self.dataset_dir.iterdir() if p.is_dir()]

        # Saving information
        self.samples: List[Sample] = []
        self.participant_calibration_data: Dict[CalibrationData] = {}

        # Tracking the number of loaded samples
        num_samples = 0

        for participant_dir in participant_dirs:

            # if self.dataset_size is not None and num_samples >= self.dataset_size:
            #     break

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

                    if self.dataset_size is not None and num_samples >= self.dataset_size:
                        break

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

                    # Detect the facial landmarks via MediaPipe
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
                    detection_results = self.face_landmarker.detect(mp_image)

                    # Compute the face bounding box based on the MediaPipe landmarks
                    try:
                        face_landmarks_proto = detection_results.face_landmarks[0]
                    except:
                        print(f"Participant {participant_id} image {items[0]} does not have a face detected.")
                        continue

                    face_landmarks = np.array([[lm.x * image.size[0], lm.y * image.size[1]] for lm in face_landmarks_proto])
                    face_bbox = np.array([
                        int(np.min(face_landmarks[:, 1])), 
                        int(np.min(face_landmarks[:, 0])), 
                        int(np.max(face_landmarks[:, 1])), 
                        int(np.max(face_landmarks[:, 0]))
                    ])

                    annotation = Annotations(
                        pog_px=np.array(items[1:3], dtype=np.float32),
                        face_bbox=face_bbox,
                        facial_landmarks_2d=face_landmarks,
                        head_pose_3d=np.array(items[15:21], dtype=np.float32).reshape(3, 2),
                        face_origin_3d=face_origin_3d,
                        face_origin_2d=face_origin_2d.flatten(),
                        gaze_target_3d=gaze_target_3d,
                        gaze_target_2d=gaze_target_2d.flatten(),
                        gaze_direction_3d=gaze_direction_3d,
                        which_eye=items[27]
                    )
            
                    annotations[items[0]] = annotation
                    num_samples += 1
 
            day_folders = [d for d in participant_dir.glob('day*') if d.is_dir()]
            for day_folder in day_folders:

                # Load the images
                images = [f for f in day_folder.glob('*.jpg') if f.is_file()]
                for image in images:
                    complete_name = f"{day_folder.name}/{image.name}"

                    if complete_name in annotations:
                        self.samples.append(
                            Sample(
                                participant_id=participant_id,
                                image_fp=image,
                                annotations=annotations[complete_name]
                            )
                        )

        # Compute the mean and standard deviation for the following information (gaze origin depth, and PoG)
        gaze_origin_depths = []
        pog_pxs = []
        for s in self.samples:
            gaze_origin_depths.append(s.annotations.face_origin_3d[2])
            pog_pxs.append(s.annotations.pog_px)

        self.gaze_origin_depth_mean = np.mean(gaze_origin_depths)
        self.gaze_origin_depth_std = np.std(gaze_origin_depths)
        self.pog_px_mean = np.mean(pog_pxs, axis=0)
        self.pog_px_std = np.std(pog_pxs, axis=0)

            
    def __getitem__(self, index: int):
        # Make a copy of the sample
        sample = copy.deepcopy(self.samples[index])

        # Create torch-compatible data
        image = Image.open(sample.image_fp)
        image_np = np.array(image)

        # Get the calibration
        calibration_data = self.participant_calibration_data[sample.participant_id]
 
        # Convert from uint8 to float32
        image_np = image_np.astype(np.float32) / 255.0

        # Draw the facial landmarks on the image
        # annotated_img = draw_landmarks_on_image(image_np, detection_results)
        # cv2.imshow('annotated_img', annotated_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Crop out the face image and resize to have standard size
        face_bbox = sample.annotations.face_bbox
        face_image_np = image_np[face_bbox[0]:face_bbox[2], face_bbox[1]:face_bbox[3]]
        face_image_np = cv2.resize(face_image_np, self.face_size, interpolation=cv2.INTER_LINEAR)

        # Visualize the face image
        # cv2.imshow('face_image', face_image_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Resize the raw input image if needed
        if self.img_size is not None:
            image_np = cv2.resize(image_np, self.img_size, interpolation=cv2.INTER_LINEAR)
            sample.annotations = resize_annotations(sample.annotations, image.size, self.img_size)
            intrinsics = resize_intrinsics(calibration_data.camera_matrix, image.size, self.img_size)
        else:
            intrinsics = calibration_data.camera_matrix
        
        # Revert the image to the correct format
        image_np = np.moveaxis(image_np, -1, 0)
        face_image_np = np.moveaxis(face_image_np, -1, 0)

        sample_dict = {
            'image': image_np,
            'face_image': face_image_np,
            'intrinsics': intrinsics,
            'dist_coeffs': calibration_data.dist_coeffs,
            'screen_R': calibration_data.monitor_rvecs.astype(np.float32),
            'screen_t': calibration_data.monitor_tvecs.astype(np.float32),
            'screen_height_mm': calibration_data.monitor_height_mm.astype(np.float32),
            'screen_height_px': calibration_data.monitor_height_px.astype(np.float32),
            'screen_width_mm': calibration_data.monitor_width_mm.astype(np.float32),
            'screen_width_px': calibration_data.monitor_width_px.astype(np.float32),
            'gaze_origin_depth_mean': self.gaze_origin_depth_mean,
            'gaze_origin_depth_std': self.gaze_origin_depth_std,
            'pog_px_mean': self.pog_px_mean,
            'pog_px_std': self.pog_px_std
        }
        sample_dict.update(asdict(sample.annotations))
        return sample_dict

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':

    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        dataset_size=2
    )
    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())