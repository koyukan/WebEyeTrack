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
from webeyetrack.utilities import create_transformation_matrix
from webeyetrack.data_protocols import GazeResult

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
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
        # import pdb; pdb.set_trace()

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

            # Convert face_origin_3d from mm to cm
            sample['face_origin_3d'] = sample['face_origin_3d'] / 10

            # Compute the error
            metrics['face_gaze_vector'].append(angle(sample['face_gaze_vector'], results.face_gaze))
            metrics['face_origin'].append(euclidean_distance(sample['face_origin_3d'], results.face_origin))
            metrics['face_origin_2d'].append(euclidean_distance(sample['face_origin_2d'], results.face_origin_2d))
            metrics['face_origin_x'].append(euclidean_distance(sample['face_origin_3d'][1], results.face_origin[1]))
            metrics['face_origin_y'].append(euclidean_distance(sample['face_origin_3d'][0], results.face_origin[0]))
            metrics['face_origin_z'].append(euclidean_distance(sample['face_origin_3d'][2], results.face_origin[2]))
            metrics['pog_px'].append(euclidean_distance(sample['pog_px'], results.pog.pog_px))
            metrics['pog_mm'].append(euclidean_distance(sample['pog_mm'], results.pog.pog_cm_s*10))

            if i % SKIP_COUNT == 0:
                # Write to the output directory
                drawn_img = visualize_differences(img.copy(), sample, results)
                cv2.imwrite(str(RUN_DIR/ 'imgs' / f'{group_name}_gaze_diff_{i}.png'), drawn_img)

                output_fp = RUN_DIR / 'imgs' / f'{group_name}_gaze_render_{i}.png'
                drawn_img = vis.render_3d_gaze_with_frame(img.copy(), results, output_fp)
                cv2.imwrite(str(output_fp), drawn_img)

                output_fp = RUN_DIR / 'imgs' / f'{group_name}_pog_render_{i}_screen.png'
                drawn_img = vis.render_pog_with_screen(
                    img.copy(), 
                    results, 
                    output_fp, 
                    screen_RT,
                    screen_height_cm=screen_height_cm,
                    screen_width_cm=screen_width_cm,
                    screen_height_px=screen_height_px,
                    screen_width_px=screen_width_px,
                    gt_pog_px=sample['pog_px'],
                )
                cv2.imwrite(str(output_fp), drawn_img)

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
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'EyeDiap'], help='Dataset to evaluate')
    # parser.add_argument('--method', type=str, required=True, choices=['blendshape', 'landmark2d', 'model-based'], help='Method to evaluate')
    args = parser.parse_args()

    eval(args)