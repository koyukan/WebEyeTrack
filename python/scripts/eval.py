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
from webeyetrack.datasets import MPIIFaceGazeDataset, GazeCaptureDataset, EyeDiapDataset
import webeyetrack.vis as vis
from webeyetrack.model_based import vector_to_pitch_yaw

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
OUTPUTS_DIR = CWD / 'outputs'
SKIP_COUNT = 10

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

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def distance(y, y_hat):
    return np.abs(y_hat - y)

def scale(y, y_hat):
    return np.abs(y / y_hat)

def angle(y, y_hat):
    return np.degrees(np.arccos(np.clip(np.dot(y, y_hat), -1.0, 1.0)))

def visualize_differences(img, sample, output):

    # Draw the facial_landmarks
    height, width = img.shape[:2]

    pitch, yaw = vector_to_pitch_yaw(sample['face_gaze_vector'])
    cv2.circle(img, (int(sample['face_origin_2d'][0]), int(sample['face_origin_2d'][1])), 5, (0, 0, 255), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=sample['face_origin_2d'][0], tdy=sample['face_origin_2d'][1], size=100)

    pitch, yaw = vector_to_pitch_yaw(output['face_gaze_vector'])
    cv2.circle(img, (int(output['face_origin_2d'][0]), int(output['face_origin_2d'][1])), 5, (255, 0, 0), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=output['face_origin_2d'][0], tdy=output['face_origin_2d'][1], size=100)

    # Draw the centers 
    for eye in ['left', 'right']:
        eye_result = output['results'].left if eye == 'left' else output['results'].right
        centroid = eye_result.origin_2d
        cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

        # Convert 3D to pitch and yaw
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        img = vis.draw_axis(img, pitch, yaw, 0, tdx=centroid[0], tdy=centroid[1], size=100)

    return img

def euclidean_distance(y, y_hat):
    return np.linalg.norm(y - y_hat)

def eval(args):
 
    # Create a dataset object
    if (args.dataset == 'MPIIFaceGaze'):
        dataset = MPIIFaceGazeDataset(
            GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
            # participants=config['datasets']['MPIIFaceGaze']['val_subjects'] + config['datasets']['MPIIFaceGaze']['train_subjects'],
            participants=[0, 1, 2],
            # img_size=[244,244],
            # face_size=[244,244],
            # dataset_size=100,
            per_participant_size=50
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

    RUN_TIMESTAMP = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    RUN_DIR = OUTPUTS_DIR / f"{RUN_TIMESTAMP}_{args.dataset}"
    os.makedirs(str(RUN_DIR), exist_ok=True)
    os.makedirs(str(RUN_DIR / 'imgs'), exist_ok=True)

    # Load the eyediap dataset
    print("FINISHED LOADING DATASET")

    metric_functions = {
        # 'depth': distance, 
        'face_gaze_vector': angle, 
        'gaze_origin': euclidean_distance, 
        'gaze_origin-x': euclidean_distance, 
        'gaze_origin-y': euclidean_distance, 
        'gaze_origin-z': euclidean_distance,
        'pog_px': euclidean_distance,
        'pog_mm': euclidean_distance
    }
    metrics = defaultdict(list)

    df = dataset.to_df()

    # Group data by participant
    for name, group in tqdm(df.groupby('participant_id')):

        # For each participant, perform calibration first by selecting 9 samples
        # by finding the closet point to the calibration points
        calib_samples = []
        for calib_point in CALIBRATION_POINTS:
            # Find the closest point to the calibration point
            distances = group.apply(lambda x: np.linalg.norm(x['pog_norm'].reshape(2) - calib_point), axis=1)
            idx = np.argmin(distances)
            sample = group.iloc[idx]
            calib_samples.append(sample)

        # Perform calibration
        # algo.calibrate(calib_samples)

        for i in tqdm(range(len(group))):

            # Get sample and load the image
            sample = df.iloc[i]
            img = np.moveaxis(sample['image'], 0, -1) * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if type(img) == type(None):
                import pdb; pdb.set_trace()

            # Process the sample
            # results = algo.process_frame(img, sample['intrinsics'])
            results = algo.process_sample(img, sample)

            output = {
                'face_origin': results.face_origin,
                'face_origin_2d': results.face_origin_2d,
                'face_origin-x': results.face_origin[1],
                'face_origin-y': results.face_origin[0],
                'face_origin-z': results.face_origin[2],
                'face_gaze_vector': results.face_gaze,
                'results': results,
            }

            # Compute the error
            actual = {
                'face_origin': sample['face_origin_3d'],
                'face_origin_2d': sample['face_origin_2d'],
                'face_origin-x': sample['face_origin_3d'][1],
                'face_origin-y': sample['face_origin_3d'][0],
                'face_origin-z': sample['face_origin_3d'][2],
                'face_gaze_vector': sample['face_gaze_vector'],
                'pog_px': sample['pog_px'],
                # 'pog_mm': sample['pog_mm']
            }
            
            for name, function in metric_functions.items():
                if name not in output or name not in actual:
                    continue
                metrics[name].append(function(actual[name], output[name]))

            if i % SKIP_COUNT == 0:
                print("Processing sample", i)
                # Write to the output directory
                drawn_img = visualize_differences(img.copy(), sample, output)
                cv2.imwrite(str(RUN_DIR/ 'imgs' / f'gaze_diff_{i}.png'), drawn_img)

                drawn_img = vis.landmark_gaze_render(img.copy(), results)
                cv2.imwrite(str(RUN_DIR/ 'imgs' / f'landmark_{i}.png'), drawn_img)

    # Generate box plots for the metrics
    df = pd.DataFrame(metrics)

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
    # print(df)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    # parser.add_argument('--method', type=str, required=True, choices=['blendshape', 'landmark2d', 'model-based'], help='Method to evaluate')
    args = parser.parse_args()

    eval(args)