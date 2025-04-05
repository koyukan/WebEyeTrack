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
from webeyetrack.utilities import vector_to_pitch_yaw, rotation_matrix_to_euler_angles, pitch_yaw_roll_to_gaze_vector

from mpiifacegaze import MPIIFaceGazeDataset
from eyediap import EyeDiapDataset
from gazecapture import GazeCaptureDataset

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

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cv2tColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def data_normalization_entry(i, sample):

    # Select the image
    frame = sample['image']
    h, w, _ = frame.shape
    facial_landmarks = sample['facial_landmarks_2d']
    # facial_landmarks = sample['facial_landmarks'][:, :2] * np.array([w, h])
    detection_results = sample['facial_detection_results']
    draw_frame = frame.copy()

    # Draw the landmarks
    draw_landmarks_simple(draw_frame, facial_landmarks)
    # draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

    # Compute the homography matrix (4 pts) from the points to a final flat rectangle
    lefttop = facial_landmarks[103]
    leftbottom = facial_landmarks[150]
    righttop = facial_landmarks[332]
    rightbottom = facial_landmarks[379]
    center = facial_landmarks[4]

    src_pts = np.array([
        lefttop,
        leftbottom,
        rightbottom,
        righttop
    ], dtype=np.float32)

    # Add padding to the points, radially away from the center
    src_direction = src_pts - center
    src_pts = src_pts + np.array([0.4, 0.2]) * src_direction

    # if draw_frame is not None:
    #     for src_pt, color in zip(src_pts, [(0,0,0), (100, 100, 100), (200, 200, 200), (255, 255, 255)]):
    #         cv2.circle(draw_frame, tuple(src_pt.astype(np.int32)), 5, color, -1)

    dst_pts = np.array([
        [0, 0],
        [0, IMG_SIZE],
        [IMG_SIZE, IMG_SIZE],
        [IMG_SIZE, 0],
    ], dtype=np.float32)

    # Compute the homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts)
    warped_face_crop = cv2.warpPerspective(frame, M, (IMG_SIZE, IMG_SIZE))

    # Apply the homography matrix to the facial landmarks
    warped_facial_landmarks = np.dot(M, np.vstack((facial_landmarks.T, np.ones((1, facial_landmarks.shape[0])))))
    warped_facial_landmarks = (warped_facial_landmarks[:2, :] / warped_facial_landmarks[2, :]).T.astype(np.int32)

    # Generate crops for the eyes and each eye separately
    top_eyes_patch = warped_facial_landmarks[151]
    bottom_eyes_patch = warped_facial_landmarks[195]
    eyes_patch = warped_face_crop[top_eyes_patch[1]:bottom_eyes_patch[1], :]
    # left_eye_in_border = warped_facial_landmarks[193]
    # right_eye_in_border = warped_facial_landmarks[417]
    # left_eye_patch = eyes_patch[:, :left_eye_in_border[0]]
    # right_eye_patch = eyes_patch[:, right_eye_in_border[0]:]

    # Reshape the eyes patch to the same size (128, 512, 3)
    eyes_patch = cv2.resize(eyes_patch, (512, 128))

    g = sample['face_gaze_vector']
    n_g = vector_to_pitch_yaw(g)
    g_pitch, g_yaw, g_roll = n_g[0], n_g[1], 0
    rt = sample['facial_rt']
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rt[:3, :3])
    h_pitch, h_yaw, h_roll = -yaw, pitch, roll

    # Create the gaze and head pose vectors
    f_g = pitch_yaw_roll_to_gaze_vector(g_pitch, g_yaw, g_roll)
    f_h = pitch_yaw_roll_to_gaze_vector(h_pitch, h_yaw, h_roll)

    oh, ow = eyes_patch.shape[:2]

    # Basic visualization for debugging purposes
    if i % 25 == 0:
        # to_visualize = cv2.equalizeHist(cv2.cv2tColor(patch, cv2.COLOR_RGB2GRAY))
        to_visualize = cv2.cvtColor(eyes_patch.copy(), cv2.COLOR_RGB2BGR)
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_g,
        #                             length=80.0, thickness=1)
        to_visualize = draw_axis(to_visualize, g_pitch, g_yaw, 0, tdx=0.5 * ow, tdy=0.4 * oh, size=100, show_xy=True)
        to_visualize = draw_axis(to_visualize, h_pitch, h_yaw, 0, tdx=0.5 * ow, tdy=0.6 * oh, size=100, color=(0, 255, 0), show_xy=True)
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
        #                             length=40.0, thickness=3, color=(0, 0, 0))
        # to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
        #                             length=40.0, thickness=1,
        #                             color=(255, 255, 255))
        cv2.imshow('frame', frame)
        cv2.imshow('draw_frame', draw_frame)
        cv2.imshow('normalized_patch', to_visualize)
        cv2.waitKey(1)

    return {
        'pixels': eyes_patch,
        'gaze_vector': f_g,
        'head_vector': f_h,

        # Take all of sample data
        **sample,
    }

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
    # parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'GazeCapture'], help='Dataset to evaluate')
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'GazeCapture'], help='Dataset to evaluate')
    args = parser.parse_args()
    
    # Generate the dataset
    generate_datasets(args)
