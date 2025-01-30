import os
import pathlib
import argparse

import numpy as np
import yaml
import tensorflow as tf
import cv2

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset, EyeDiapDataset

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
os.makedirs(GENERATED_DATASET_DIR, exist_ok=True)

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, np.ndarray):  # Convert NumPy array to bytes
        # value = value.tobytes()
        value = cv2.imencode('.png', value)[1].tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(image, gaze_vector):
    """Creates a serialized example for TFRecord."""
    feature = {
        'image': _bytes_feature(image),
        'gaze_vector': _float_feature(gaze_vector),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def eval(args):
 
    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        train_dataset = MPIIFaceGazeDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
            participants=config['datasets']['MPIIFaceGaze']['train_subjects'],
            face_size=[128,128],
            # per_participant_size=100
        )
        val_dataset = MPIIFaceGazeDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
            # participants=config['datasets']['MPIIFaceGaze']['val_subjects'],
            participants=[14],
            face_size=[128,128],
            # per_participant_size=100
        )
    elif (args.dataset == 'EyeDiap'):
        train_dataset = EyeDiapDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
            participants=1,
            # dataset_size=20,
            # per_participant_size=10,
            # video_type='hd'
            video_type='vga',
            frame_skip_rate=5
        )
        val_dataset = None
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")

    # output_path = GENERATED_DATASET_DIR / f'train_{args.dataset}_blazegaze.tfrecord'
    # with tf.io.TFRecordWriter(str(output_path)) as writer:
    #     for sample in train_dataset:
    #         face_image = sample['face_image']
    #         face_image = np.moveaxis(face_image, 0, -1)
    #         face_image = (face_image * 255).astype(np.uint8)
    #         assert face_image.shape == (128, 128, 3)
    #         gaze_vector = sample['face_gaze_vector']
    #         example = serialize_example(face_image, gaze_vector)
    #         writer.write(example)

    output_path = GENERATED_DATASET_DIR / f'val_{args.dataset}_blazegaze_p14.tfrecord'
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for sample in val_dataset:
            face_image = sample['face_image']
            face_image = np.moveaxis(face_image, 0, -1)
            face_image = (face_image * 255).astype(np.uint8)
            assert face_image.shape == (128, 128, 3)
            gaze_vector = sample['face_gaze_vector']
            example = serialize_example(face_image, gaze_vector)
            writer.write(example)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()
    eval(args)