import os
import pathlib
import argparse
import datetime
import json

import numpy as np
import yaml
import tensorflow as tf
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset, EyeDiapDataset
from webeyetrack.utilities import vector_to_pitch_yaw, rotation_matrix_to_euler_angles

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
os.makedirs(GENERATED_DATASET_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

def serialize_example(sample):
    """Creates a serialized example for TFRecord."""

    # Include the face image
    face_image = sample['face_image']
    face_image = np.moveaxis(face_image, 0, -1)
    face_image = (face_image * 255).astype(np.uint8)
    assert face_image.shape == (128, 128, 3)

    # Gaze vector
    gaze_vector = sample['face_gaze_vector']

    # Convert gaze vector to pitch and yaw
    pitch, yaw = vector_to_pitch_yaw(gaze_vector, degrees=False)
    gaze_pitch_yaw = np.array([-pitch, yaw]) # Add negative sign to pitch due to conversion

    # Head pose vector (only take the rotation matrix)
    facial_rt = sample['facial_rt'] # (4, 4)
    head_rotation = facial_rt[:3, :3] # (3, 3)

    # Convert head_rotation from 3x3 to pitch, yaw, roll
    pitch, yaw, roll = rotation_matrix_to_euler_angles(head_rotation, degrees=False)
    head_pitch_yaw_roll = np.array([pitch, yaw, roll])

    feature = {
        'image': _bytes_feature(face_image),
        'gaze_vector': _float_feature(gaze_pitch_yaw),
        'head_rotation': _float_feature(head_pitch_yaw_roll),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def load_datasets(args):

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
            participants=config['datasets']['MPIIFaceGaze']['val_subjects'],
            # participants=[14],
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

    return train_dataset, val_dataset

# def feature_mean_std(args):

#     # Load the dataset
#     train_dataset, val_dataset = load_datasets(args)

#     # Initialize accumulators
#     image_sum = 0
#     image_sq_sum = 0
#     num_images = 0

#     gaze_mean = 0
#     gaze_sq_sum = 0

#     head_rotation_mean = 0
#     head_rotation_sq_sum = 0

#     for sample in tqdm(train_dataset, total=len(train_dataset)):
#         face_image = sample['face_image']  # Shape: (H, W, C)
#         face_gaze_vector = sample['face_gaze_vector']
#         facial_rt = sample['facial_rt']
#         head_rotation = facial_rt[:3, :3]

#         # Ensure the data is correct for the model
#         face_image = np.moveaxis(face_image, 0, -1)
#         assert face_image.shape == (128, 128, 3)

#         # Convert image to float32 for precision
#         face_image = face_image.astype(np.float32) / 255.0  # Normalize between 0 and 1 if not already

#         # Accumulate sums for mean calculation
#         image_sum += np.mean(face_image, axis=(0, 1))  # Compute per-channel mean
#         image_sq_sum += np.mean(face_image ** 2, axis=(0, 1))  # Compute per-channel squared mean

#         gaze_mean += np.mean(face_gaze_vector)
#         gaze_sq_sum += np.mean(face_gaze_vector ** 2)

#         head_rotation_mean += np.mean(head_rotation)
#         head_rotation_sq_sum += np.mean(head_rotation ** 2)

#         num_images += 1

#     # Compute final means
#     image_mean = image_sum / num_images
#     image_std = np.sqrt((image_sq_sum / num_images) - (image_mean ** 2))  # Standard deviation formula

#     gaze_mean /= num_images
#     gaze_std = np.sqrt((gaze_sq_sum / num_images) - (gaze_mean ** 2))

#     head_rotation_mean /= num_images
#     head_rotation_std = np.sqrt((head_rotation_sq_sum / num_images) - (head_rotation_mean ** 2))

#     # Print results
#     print(f"Image mean: {image_mean}, Image std: {image_std}")
#     print(f"Gaze mean: {gaze_mean}, Gaze std: {gaze_std}")
#     print(f"Head rotation mean: {head_rotation_mean}, Head rotation std: {head_rotation_std}")

#     # Save into a JSON file
#     mean_std = {
#         'image_mean': image_mean.tolist(),
#         'image_std': image_std.tolist(),
#         'gaze_mean': gaze_mean,
#         'gaze_std': gaze_std,
#         'head_rotation_mean': head_rotation_mean,
#         'head_rotation_std': head_rotation_std,
#     }
#     with open(GENERATED_DATASET_DIR / f'{args.dataset}_blazegaze_mean_std_{TIMESTAMP}.json', 'w') as f:
#         json.dump(mean_std, f, indent=4)

def feature_mean_std(args):

    # Load the dataset
    train_dataset, val_dataset = load_datasets(args)

    # Initialize accumulators for per-channel mean and std
    image_sum = np.zeros(3)  # For R, G, B channels
    image_sq_sum = np.zeros(3)
    num_pixels = 0

    gaze_sum = np.zeros(2)  # Assuming (pitch, yaw)
    gaze_sq_sum = np.zeros(2)

    head_rotation_sum = np.zeros(3)  # Assuming (pitch, yaw, roll)
    head_rotation_sq_sum = np.zeros(3)

    for sample in tqdm(train_dataset, total=len(train_dataset)):
        face_image = sample['face_image']  # Shape: (H, W, C)
        face_gaze_vector = sample['face_gaze_vector']
        # head_rotation = sample['head_rotation']

        # Ensure image format is correct
        face_image = np.moveaxis(face_image, 0, -1) # Already float32
        assert face_image.shape == (128, 128, 3)

        # Accumulate sums for mean calculation (per channel)
        image_sum += np.sum(face_image, axis=(0, 1))  # Sum over height & width for each channel
        image_sq_sum += np.sum(face_image ** 2, axis=(0, 1))  # Sum of squared values

        # Compute the number of pixels
        num_pixels += np.prod(face_image.shape[:2])

        # Accumulate gaze and head rotation values
        # Convert gaze vector to pitch and yaw
        pitch, yaw = vector_to_pitch_yaw(face_gaze_vector, degrees=False)
        gaze_pitch_yaw = np.array([-pitch, yaw]) # Add negative sign to pitch due to conversion
        gaze_sum += gaze_pitch_yaw
        gaze_sq_sum += gaze_pitch_yaw ** 2

        # head_rotation_sum += head_rotation
        # head_rotation_sq_sum += head_rotation ** 2

    # Compute the total number of samples
    num_samples = len(train_dataset)

    # Compute per-channel mean and std
    image_mean = image_sum / num_pixels
    image_std = np.sqrt((image_sq_sum / num_pixels) - (image_mean ** 2))  # Standard deviation formula

    # Compute mean/std for gaze and head rotation
    gaze_mean = gaze_sum / num_samples
    gaze_std = np.sqrt((gaze_sq_sum / num_samples) - (gaze_mean ** 2))

    # head_rotation_mean = head_rotation_sum / num_samples
    # head_rotation_std = np.sqrt((head_rotation_sq_sum / num_samples) - (head_rotation_mean ** 2))

    # Print results
    print(f"Image mean: {image_mean}, Image std: {image_std}")
    print(f"Gaze mean: {gaze_mean}, Gaze std: {gaze_std}")
    # print(f"Head rotation mean: {head_rotation_mean}, Head rotation std: {head_rotation_std}")

    # Save into a JSON file
    mean_std = {
        'image_mean': image_mean.tolist(),
        'image_std': image_std.tolist(),
        'gaze_mean': gaze_mean.tolist(),
        'gaze_std': gaze_std.tolist(),
        # 'head_rotation_mean': head_rotation_mean.tolist(),
        # 'head_rotation_std': head_rotation_std.tolist(),
    }
    with open(GENERATED_DATASET_DIR / f'{args.dataset}_blazegaze_mean_std_{TIMESTAMP}.json', 'w') as f:
        json.dump(mean_std, f, indent=4)


def generate_datasets(args):

    # Load the dataset
    train_dataset, val_dataset = load_datasets(args)

    output_path = GENERATED_DATASET_DIR / f'train_{args.dataset}_blazegaze_{TIMESTAMP}.tfrecord'
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for sample in train_dataset:
            example = serialize_example(sample)
            writer.write(example)

    output_path = GENERATED_DATASET_DIR / f'val_{args.dataset}_blazegaze_{TIMESTAMP}.tfrecord'
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for sample in val_dataset:
            example = serialize_example(sample)
            writer.write(example)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()
    
    # Compute the input features mean and std to normalize the data
    feature_mean_std(args)
    
    # Generate the dataset
    # generate_datasets(args)