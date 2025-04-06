import os
import sys
import pathlib
from collections import defaultdict
import argparse
import json

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack import WebEyeTrack, WebEyeTrackConfig
from webeyetrack.constants import GIT_ROOT
import webeyetrack.vis as vis
from webeyetrack.model_based import vector_to_pitch_yaw, compute_pog
from webeyetrack.utilities import create_transformation_matrix
from webeyetrack.data_protocols import GazeResult


CWD = pathlib.Path(__file__).parent
sys.path.append(str(CWD.parent / 'preprocessing'))
from mpiifacegaze import MPIIFaceGazeDataset
from gazecapture import GazeCaptureDataset

FILE_DIR = CWD.parent
OUTPUTS_DIR = CWD / 'outputs'
SKIP_COUNT = 100
# SKIP_COUNT = 2
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

# Load the GazeCapture participant IDs
with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
    GAZE_CAPTURE_IDS = json.load(f)

TOTAL_DATASET = 100
PER_PARTICIPANT_SIZE = 5

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def distance(y, y_hat):
    return np.abs(y_hat - y)

def scale(y, y_hat):
    return np.abs(y / y_hat)

def angle(y, y_hat):
    return np.degrees(np.arccos(np.clip(np.dot(y, y_hat), -1.0, 1.0)))

def euclidean_distance(y, y_hat):
    return np.linalg.norm(y - y_hat)

def visualize_differences(img, sample, results: GazeResult):

    # Draw the facial_landmarks
    pitch, yaw = vector_to_pitch_yaw(sample['face_gaze_vector'])
    cv2.circle(img, (int(sample['face_origin_2d'][0]), int(sample['face_origin_2d'][1])), 5, (0, 0, 255), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=sample['face_origin_2d'][0], tdy=sample['face_origin_2d'][1], size=100)

    pitch, yaw = vector_to_pitch_yaw(results.face_gaze)
    cv2.circle(img, (int(results.face_origin_2d[0]), int(results.face_origin_2d[1])), 5, (255, 0, 0), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=results.face_origin_2d[0], tdy=results.face_origin_2d[1], size=100)

    # Draw the centers 
    for eye in ['left', 'right']:
        eye_result = results.left if eye == 'left' else results.right
        centroid = eye_result.origin_2d
        cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

        # Convert 3D to pitch and yaw
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        img = vis.draw_axis(img, pitch, yaw, 0, tdx=centroid[0], tdy=centroid[1], size=100)

    return img

def eval(args):
 
    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        dataset = MPIIFaceGazeDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
            participants=[x for x in range(14)],
            dataset_size=TOTAL_DATASET,
            per_participant_size=PER_PARTICIPANT_SIZE
        )
    elif (args.dataset == 'GazeCapture'):
        dataset = GazeCaptureDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['GazeCapture']['path']),
            participants=GAZE_CAPTURE_IDS,
            dataset_size=TOTAL_DATASET,
            per_participant_size=PER_PARTICIPANT_SIZE
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    RUN_TIMESTAMP = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    RUN_DIR = OUTPUTS_DIR / f"{RUN_TIMESTAMP}_{args.dataset}"
    os.makedirs(str(RUN_DIR), exist_ok=True)
    os.makedirs(str(RUN_DIR / 'imgs'), exist_ok=True)

    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")

    metrics = defaultdict(list)

    # Obtain the sample dataframe
    meta_df = dataset.get_samples_meta_df()

    # Group data by participant
    for group_name, group in tqdm(meta_df.groupby('participant_id')):

        # Create the WebEyeTrack object
        wet = WebEyeTrack()

        # For each participant, perform calibration first by selecting 9 samples
        # by finding the closet point to the calibration points
        # calib_samples = []
        # for calib_point in CALIBRATION_POINTS:
        #     # Find the closest point to the calibration point
        #     distances = group.apply(lambda x: np.linalg.norm(x['pog_norm'].reshape(2) - calib_point), axis=1)
        #     idx = np.argmin(distances)
        #     sample = group.iloc[idx]
        #     calib_samples.append(sample)

        # Perform calibration
        # algo.calibrate(calib_samples)

        # for i in tqdm(range(len(group))):
        for i, meta_data in tqdm(group.iterrows(), total=len(group)):

            # Obtain the sample
            sample = dataset.__getitem__(meta_data.name)

            # Get sample and load the image
            img = sample['image']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if type(img) == type(None):
                print(f"Image is None for {sample['image_fp']}")
                continue

            # Show the image
            cv2.imshow('img', img)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

            # Process the sample
            pog_cm = wet.step(
                img,
                sample['facial_landmarks'], 
                sample['facial_rt'], 
            )

            # Compute the POG error in euclidean distance (cm)
            pog_error = euclidean_distance(sample['pog_cm'], pog_cm)

            # Store the metrics
            metrics['pog_error'].append(pog_error)
            # metrics['participant_id'].append(group_name)

    # Generate box plots for the metrics
    df = pd.DataFrame(metrics)
    if df.empty:
        return

    # Remove outliers via IQR for all metrics 
    for name in metrics.keys():
        q1 = df[name].quantile(0.10)
        q3 = df[name].quantile(0.90)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        df = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]

    # Calculate mean and std for each metric
    num_metrics = len(metrics.keys())
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 6))
    fig.suptitle('Metrics Evaluation', fontsize=16)

    for i, name in enumerate(metrics.keys()):
        mean = df[name].mean()
        std = df[name].std()
          
        # Add mean and std as text in the figure
        if num_metrics > 1:
            # Plot the boxplot for each metric
            sns.boxplot(data=df, y=name, ax=axes[i])
            axes[i].text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                        transform=axes[i].transAxes, 
                        verticalalignment='top')
            
            # Set the title with mean and std
            axes[i].set_title(f'{name.capitalize()}\nMean: {mean:.2f}, Std: {std:.2f}')
        else:
            # Plot the boxplot for each metric
            sns.boxplot(data=df, y=name, ax=axes)
            axes.text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                        transform=axes.transAxes, 
                        verticalalignment='top')
            
            # Set the title with mean and std
            axes.set_title(f'{name.capitalize()}\nMean: {mean:.2f}, Std: {std:.2f}')

        
    # plt.tight_layout()
    # plt.show()
    plt.savefig(str(RUN_DIR / 'metrics.png'))

    print(df.describe())
    df.to_excel(str(RUN_DIR / 'metrics.xlsx'))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'GazeCapture'], help='Dataset to evaluate')
    args = parser.parse_args()

    eval(args)