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
    return np.degrees(np.arccos(np.clip(np.dot(y, y_hat), -1.0, 1.0)))

def visualize_gaze_vectors(sample, output):
    
    # Draw the gaze vectors
    import pdb; pdb.set_trace()
    img = np.moveaxis(sample['image'], 0, -1)
    gt_gaze = vis.draw_gaze_direction(img, sample['face_origin_2d'], sample['gaze_target_2d'], color=(0, 0, 255))
    gaze_target_3d_semi = sample['face_origin_3d'] + output['face_gaze_vector'] * 100
    gaze_target_2d, _ = cv2.projectPoints(
        gaze_target_3d_semi, 
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32),
        sample['intrinsics'], 
        sample['dist_coeffs'],
    )
    gt_pred_gaze = vis.draw_gaze_direction(
        gt_gaze,
        sample['face_origin_2d'],
        gaze_target_2d.flatten(),
        color=(255, 0, 0)
    )

    return gt_pred_gaze
    # plt.imshow(gt_pred_gaze)
    # plt.show()

def visualize_gaze_2d_origin(img, point_a, point_b):
    img = np.moveaxis(img, 0, -1)
    # Draw the gaze origin from the sample in 2D
    draw_img = vis.draw_gaze_origin(img, point_a, color=(255, 0, 0)) 
    # Draw the gaze origin from the output in 2D
    draw_img = vis.draw_gaze_origin(draw_img, point_b, color=(0, 0, 255))
    return draw_img

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
        participants=config['datasets']['MPIIFaceGaze']['val_subjects'],
        # img_size=[244,244],
        # face_size=[244,244],
        dataset_size=10
    )

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
        gaze_result = algo.process_sample(sample)
        # except Exception as e:
        #     print(e)
        #     continue

        output = {
            'depth': gaze_result.face_origin[2],
            'gaze_origin': gaze_result.face_origin,
            'gaze_origin_2d': gaze_result.face_origin_2d,
            'face_gaze_vector': gaze_result.face_gaze,
            # 'gaze_origin-x': gaze_result.face_origin[1],
            # 'gaze_origin-y': gaze_result.face_origin[0],
            # 'gaze_origin-z': gaze_result.face_origin[2],
        }

        # # Separate the xyz gaze origin
        # output['gaze_origin-x'] = output['gaze_origin'][0]
        # output['gaze_origin-y'] = output['gaze_origin'][1]
        # output['gaze_origin-z'] = output['gaze_origin'][2]

        # Compute the error
        actual = {
            'depth': sample['face_origin_3d'][2],
            'gaze_origin': sample['face_origin_3d'],
            'gaze_origin-x': sample['face_origin_3d'][1],
            'gaze_origin-y': sample['face_origin_3d'][0],
            'gaze_origin-z': sample['face_origin_3d'][2],
            'face_gaze_vector': sample['gaze_direction_3d'],
            'pog_px': sample['pog_px'],
            'pog_mm': sample['pog_mm']
        }
        
        for name, function in metric_functions.items():
            if name not in output or name not in actual:
                continue
            metrics[name].append(function(actual[name], output[name]))

        # Write to the output directory
        img = visualize_gaze_vectors(sample, output)
        cv2.imwrite(str(OUTPUTS_DIR / f'gaze_vectors_{i}.png'), img)
        # img = visualize_gaze_2d_origin(sample['image'], sample['face_origin_2d'], gaze_result.face_origin_2d)
        cv2.imwrite(str(OUTPUTS_DIR / f'gaze_origin_{i}.png'), img)

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

        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    eval()