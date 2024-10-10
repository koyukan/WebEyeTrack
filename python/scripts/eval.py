import pathlib
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset
from webeyetrack.pipelines import FLGE
import webeyetrack.vis as vis
import webeyetrack.core as core

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
OUTPUTS_DIR = CWD / 'outputs'

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def distance(y, y_hat):
    return np.abs(y_hat - y)

def scale(y, y_hat):
    return np.abs(y / y_hat)

def angle(y, y_hat):
    # import pdb; pdb.set_trace()
    return np.degrees(np.arccos(np.clip(np.dot(y, y_hat), -1.0, 1.0)))

def visualize_differences(sample, output):
    
    # Draw the gaze vectors
    img = cv2.cvtColor(np.moveaxis(sample['image'], 0, -1) * 255, cv2.COLOR_RGB2BGR)

    # Draw the facial_landmarks
    height, width = img.shape[:2]

    pitch, yaw = core.vector_to_pitch_yaw(sample['gaze_direction_3d'])
    cv2.circle(img, (int(sample['face_origin_2d'][0]), int(sample['face_origin_2d'][1])), 5, (0, 0, 255), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=sample['face_origin_2d'][0], tdy=sample['face_origin_2d'][1], size=100)

    pitch, yaw = core.vector_to_pitch_yaw(output['face_gaze_vector'])
    cv2.circle(img, (int(output['face_origin_2d'][0]), int(output['face_origin_2d'][1])), 5, (255, 0, 0), -1)
    img = vis.draw_axis(img, pitch, yaw, 0, tdx=output['face_origin_2d'][0], tdy=output['face_origin_2d'][1], size=100)

    # Draw the centers 
    for eye in ['left', 'right']:
        eye_result = output['results'].left if eye == 'left' else output['results'].right
        centroid = eye_result.origin_2d
        cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

        # Convert 3D to pitch and yaw
        pitch, yaw = core.vector_to_pitch_yaw(eye_result.direction)
        img = vis.draw_axis(img, pitch, yaw, 0, tdx=centroid[0], tdy=centroid[1], size=100)

    return img

def euclidean_distance(y, y_hat):
    return np.linalg.norm(y - y_hat)

def eval():

    # Create pipeline
    algo = FLGE(
        str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task')
    )
    
    # Create a dataset object
    dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        participants=config['datasets']['MPIIFaceGaze']['val_subjects'] + config['datasets']['MPIIFaceGaze']['train_subjects'],
        # participants=[1,2],
        # img_size=[244,244],
        # face_size=[244,244],
        # dataset_size=10
    )

    print("FINISHED LOADING DATASET")

    metric_functions = {
        'depth': distance, 
        'face_gaze_vector': angle, 
        'gaze_origin': euclidean_distance, 
        'gaze_origin-x': euclidean_distance, 
        'gaze_origin-y': euclidean_distance, 
        'gaze_origin-z': euclidean_distance,
        'pog_px': euclidean_distance,
        'pog_mm': euclidean_distance
    }
    metrics = defaultdict(list)
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

        # Process the sample
        # try:
        # results = algo.process_sample(sample)
        img = cv2.cvtColor(np.moveaxis(sample['image'], 0, -1) * 255, cv2.COLOR_RGB2BGR)
        results = algo.process_frame(img, sample['intrinsics'])
        # except Exception as e:
        #     print(e)
        #     continue

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
            'face_gaze_vector': sample['gaze_direction_3d'],
            'pog_px': sample['pog_px'],
            'pog_mm': sample['pog_mm']
        }
        
        for name, function in metric_functions.items():
            if name not in output or name not in actual:
                continue
            metrics[name].append(function(actual[name], output[name]))

        if i % 100 == 0:
            # Write to the output directory
            # img = visualize_gaze_vectors(sample, output)
            # cv2.imwrite(str(OUTPUTS_DIR / f'gaze_vectors_{i}.png'), img)
            # img = visualize_gaze_2d_origin(sample['image'], sample['face_origin_2d'], results.face_origin_2d)
            # cv2.imwrite(str(OUTPUTS_DIR / f'gaze_origin_{i}.png'), img)
            img = visualize_differences(sample, output)
            cv2.imwrite(str(OUTPUTS_DIR / 'imgs' / f'gaze_diff_{i}.png'), img)
            # img = vis.landmark_gaze_render(img, results)
            # cv2.imwrite(str(OUTPUTS_DIR / f'landmark_{i}.png'), img)

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
    plt.savefig(str(OUTPUTS_DIR / 'metrics.png'))

    print(df.describe())
    # print(df)

if __name__ == '__main__':
    eval()