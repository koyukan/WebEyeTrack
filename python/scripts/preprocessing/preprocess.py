import os
import pathlib
import argparse
import datetime
from collections import defaultdict
import json

import numpy as np
import yaml
import tensorflow as tf
import cv2
from tqdm import tqdm
import h5py

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.vis import draw_axis, draw_landmarks_simple
from webeyetrack.constants import GIT_ROOT

from mpiifacegaze import MPIIFaceGazeDataset
from eyediap import EyeDiapDataset
from gazecapture import GazeCaptureDataset
from utils import data_normalization_entry

CWD = pathlib.Path(__file__).parent
SCRIPTS_DIR = CWD.parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
os.makedirs(GENERATED_DATASET_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Load the GazeCapture participant IDs
with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
    GAZE_CAPTURE_IDS = json.load(f)

with open(SCRIPTS_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

IMG_SIZE = 512

def load_datasets(args):

    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        person_datasets = {}
        total_participants = config['datasets']['MPIIFaceGaze']['train_subjects'] + config['datasets']['MPIIFaceGaze']['val_subjects']
        for participant in total_participants: 
            dataset = MPIIFaceGazeDataset(
                GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
                participants=[participant],
                # per_participant_size=100,
                # face_size=[128,128],
                # per_participant_size=10
            )
            person_datasets[participant] = dataset
    elif (args.dataset == 'GazeCapture'):
        person_datasets = {}
        for participant in tqdm(GAZE_CAPTURE_IDS, desc='Participants'):
            dataset = GazeCaptureDataset(
                GIT_ROOT / pathlib.Path(config['datasets']['GazeCapture']['path']),
                participants=[participant],
                # dataset_size=20,
                # per_participant_size=10,
            )
            person_datasets[participant] = dataset
    elif (args.dataset == 'EyeDiap'):
        person_datasets = {}
        total_participants = list(range(1, 17)) # 1-16
        for participant in tqdm(total_participants, desc='Participants'):
            dataset = EyeDiapDataset(
                GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
                participants=[participant],
                # dataset_size=200,
                # per_participant_size=10,
            )
            person_datasets[participant] = dataset
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")
    return person_datasets

def generate_datasets(args):
    
    # Create output path
    output_path = GENERATED_DATASET_DIR / f'{args.dataset}_{TIMESTAMP}.h5'
    print(f"Generating {output_path}")

    # Load the dataset
    person_datasets = load_datasets(args)

    for person_id, person_dataset in tqdm(person_datasets.items(), total=len(person_datasets), desc='Person ID'):
        # Prepare methods to organize per-entry outputs
        to_write = defaultdict(list)

        for i, sample in tqdm(enumerate(person_dataset), total=len(person_dataset), desc='Sample'):
            
            # Perform data normalization
            try:
                processed_entry = data_normalization_entry(i, sample)
            except Exception as e:
                print(f"Error processing entry {i}: {e}")
                continue

            to_write['pixels'].append(processed_entry['pixels'])
            # to_write['labels'].append(np.concatenate([
            #     processed_entry['gaze_vector'],
            #     processed_entry['head_vector'],
            # ]))
            to_write['gaze_vector'].append(processed_entry['gaze_vector'])

            # Include head pose information
            to_write['face_origin_3d'].append(processed_entry['face_origin_3d'])
            to_write['face_origin_2d'].append(processed_entry['face_origin_2d'])
            to_write['head_vector'].append(processed_entry['head_vector'])
            
            # Include 2D Gaze information
            to_write['pog_px'].append(processed_entry['pog_px'])
            to_write['pog_norm'].append(processed_entry['pog_norm'])
            to_write['pog_cm'].append(processed_entry['pog_cm'])
            to_write['screen_height_cm'].append(processed_entry['screen_height_cm'])
            to_write['screen_width_cm'].append(processed_entry['screen_width_cm'])
            to_write['screen_height_px'].append(processed_entry['screen_height_px'])
            to_write['screen_width_px'].append(processed_entry['screen_width_px'])

        if len(to_write) == 0:
            continue

        # Cast to numpy arrays
        # for key, values in to_write.items():
        #     to_write[key] = np.array(values)
        #     print(f"Key: {key}, Shape: {to_write[key].shape}")
            
        # Write to HDF
        with h5py.File(output_path, 'a' if os.path.isfile(output_path) else 'w') as f:
            if str(person_id) in f:
                del f[person_id]
            group = f.create_group(str(person_id))
            for key, values in tqdm(to_write.items(), total=len(to_write), desc='Writing'):
                group.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'GazeCapture', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()
    
    # Generate the dataset
    generate_datasets(args)
