import os
import pathlib
import argparse
import datetime

import numpy as np
import yaml
import tensorflow as tf
import cv2
from tqdm import tqdm
import h5py

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.constants import GIT_ROOT
from webeyetrack.utilities import vector_to_pitch_yaw, rotation_matrix_to_euler_angles

from mpiifacegaze import MPIIFaceGazeDataset

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
os.makedirs(GENERATED_DATASET_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TO_WRITE = []

def load_datasets(args):

    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        person_datasets = {}
        total_participants = config['datasets']['MPIIFaceGaze']['train_subjects'] + config['datasets']['MPIIFaceGaze']['val_subjects']
        for participant in total_participants: 
            dataset = MPIIFaceGazeDataset(
                GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
                participants=[participant],
                face_size=[128,128],
                per_participant_size=100
            )
            person_datasets[participant] = dataset
    # elif (args.dataset == 'EyeDiap'):
    #     dataset = EyeDiapDataset(
    #         GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
    #         participants=1,
    #         # dataset_size=20,
    #         # per_participant_size=10,
    #         # video_type='hd'
    #         video_type='vga',
    #         frame_skip_rate=5
    #     )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")
    return person_datasets

def data_normalization_entry(sample):
    ...

def generate_datasets(args):

    # Prepare methods to organize per-entry outputs
    to_write = {}
    def add(key, value):  # noqa
        if key not in to_write:
            to_write[key] = [value]
        else:
            to_write[key].append(value)

    # Load the dataset
    person_datasets = load_datasets(args)

    # Create output path
    output_path = GENERATED_DATASET_DIR / f'{args.dataset}_{TIMESTAMP}.h5'

    with h5py.File(output_path, 'w') as f:
        for person_id, person_dataset in person_datasets:
            group = f.create_group(person_id)
            for sample in person_dataset:
                
                # Perform data normalization
                processed_entry = data_normalization_entry(sample)

                add('face_image', processed_entry['face_image'])
                
                # Write the sample to the group
                for key, value in to_write.items():
                    group.create_dataset(
                        key, data=value,
                        chunks=(
                            tuple([1] + list[value.shape[1:]])
                            if isinstance(value, np.ndarray)
                            else None
                        ),
                        compression='lzf',
                    )

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()
    
    # Generate the dataset
    generate_datasets(args)