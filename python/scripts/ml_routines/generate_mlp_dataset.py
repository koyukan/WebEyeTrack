import os
import pathlib
from collections import defaultdict
import argparse

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack import WebEyeTrack
from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset, EyeDiapDataset
from webeyetrack.utilities import create_transformation_matrix
from webeyetrack.data_protocols import GazeResult

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
CALIBRATION_POINTS = np.array([ # 9 points
    [0.5, 0.5],
    [0.1, 0.1],
    [0.1, 0.9],
    [0.9, 0.1],
    [0.9, 0.9],
    [0.5, 0.1],
    [0.5, 0.9],
    [0.1, 0.5],
    [0.9, 0.5]
])
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
os.makedirs(GENERATED_DATASET_DIR, exist_ok=True)

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def eval(args):
 
    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        dataset = MPIIFaceGazeDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
            participants=config['datasets']['MPIIFaceGaze']['val_subjects'] + config['datasets']['MPIIFaceGaze']['train_subjects'],
            # participants=[5, 6, 7, 8, 9],
            # participants=[5],
            # img_size=[244,244],
            # face_size=[244,244],
            # dataset_size=100,
            # per_participant_size=500
            # per_participant_size=5
        )
    elif (args.dataset == 'EyeDiap'):
        dataset = EyeDiapDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
            participants=1,
            # dataset_size=20,
            # per_participant_size=10,
            # video_type='hd'
            video_type='vga',
            frame_skip_rate=5
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Create pipeline
    algo = WebEyeTrack(
        str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'),
    )

    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")

    # Obtain the sample dataframe
    meta_df = dataset.get_samples_meta_df()

    new_dataset = defaultdict(list)

    # Group data by participant
    for group_name, group in tqdm(meta_df.groupby('participant_id')):

        first_sample_meta_df = group.iloc[0]
        first_sample = dataset.__getitem__(first_sample_meta_df.name)

        screen_RT = np.linalg.inv(create_transformation_matrix(
            scale=1,
            rotation=first_sample['screen_R'],
            translation=first_sample['screen_t']/10
        ))
        screen_width_cm = first_sample['screen_width_mm'].flatten()[0]/10
        screen_height_cm = first_sample['screen_height_mm'].flatten()[0]/10
        screen_width_px = int(first_sample['screen_width_px'].flatten()[0])
        screen_height_px = int(first_sample['screen_height_px'].flatten()[0])

        # Correcting the screen_RT by multiplying the X axis by -1
        screen_RT[0, :] = screen_RT[0, :] * -1

        # Update the configurations
        algo.config(
            face_width_cm=None, # Reset the head scale estimation
            intrinsics=first_sample['intrinsics'],
            screen_RT=screen_RT,
            screen_width_cm=screen_width_cm,
            screen_height_cm=screen_height_cm,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
        )

        # for i in tqdm(range(len(group))):
        for i, meta_data in tqdm(group.iterrows(), total=len(group)):

            # Obtain the sample
            sample = dataset.__getitem__(meta_data.name)

            # Convert face_origin_3d from mm to cm
            sample['face_origin_3d'] = sample['face_origin_3d'] / 10

            # Get sample and load the image
            img = np.moveaxis(sample['image'], 0, -1) * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if type(img) == type(None):
                print(f"Image is None for {sample['image_fp']}")
                continue

            # Update the last configs
            algo.config(
                frame_height=img.shape[0],
                frame_width=img.shape[1],
            )

            # Extract the necessary input for the algo
            facial_landmarks = sample['facial_landmarks']
            face_rt = sample['facial_rt']
            face_blendshapes = sample['face_blendshapes']

            # Process the sample
            results = algo.step(facial_landmarks, face_rt, face_blendshapes)

            import pdb; pdb.set_trace()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()
    eval(args)