import pathlib
import argparse
import json

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import h5py
import yaml

import tensorflow as tf

from webeyetrack.constants import GIT_ROOT

from preprocess import load_datasets

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

def dataset_generator(h5_file, participants=None, max_size=None):
    def _gen(participants=participants, max_size=max_size):
        counter = 0
        with h5py.File(h5_file, 'r') as f:
            if participants == None:
                participants = list(f.keys())
            for pid in participants:
                group = f[str(pid)]
                total = group['pixels'].shape[0]
                for i in range(total):
                    
                    pog_norm = group['pog_norm'][i]

                    # If any value is > 1.0 or < 0.0, skip this sample
                    if np.any(pog_norm > 1.0) or np.any(pog_norm < 0.0):
                        continue

                    yield {
                        'pog_norm': group['pog_norm'][i][:2].astype(np.float32),
                        'pog_px': group['pog_px'][i][:2].astype(np.float32)
                    }
                    counter += 1

                    if max_size is not None and counter >= max_size:
                        print(f"Reached max size of {max_size} samples.")
                        return

            return

    return _gen()

def visualize_output_space(dataset):

    # Collect all normalized PoG values
    all_pogs = []
    to_cm = False

    print("Collecting normalized PoG values...")
    for sample in tqdm(dataset, desc="Samples"):
        pog = sample['pog_norm']
        if pog is not None:
            all_pogs.append(pog)

    # Concatenate into a single array
    all_pogs = np.stack(all_pogs, axis=0)
    x_vals, y_vals = all_pogs[:, 0]-0.5, all_pogs[:, 1]-0.5

    # 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # extent = [0, 1, 0, 1]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(
        heatmap.T,
        extent=extent,
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.colorbar(label='Frequency')
    xlabel = 'X [cm]' if to_cm else 'X (Normalized)'
    ylabel = 'Y [cm]' if to_cm else 'Y (Normalized)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("2D Heatmap of Gaze Targets")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGaze', 'GazeCapture', 'EyeDiap'], help='Dataset to evaluate')
    args = parser.parse_args()

    if args.dataset == 'GazeCapture':
        with open(CWD.parent / 'ml_routines' / 'configs' / 'train' / 'maml_gazecapture_config.yml', 'r') as f:
            config = yaml.safe_load(f)
        with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
            GAZE_CAPTURE_IDS = json.load(f)
        h5_file = GENERATED_DATASET_DIR / 'GazeCapture_entire.h5'
        dataset = dataset_generator(h5_file, participants=GAZE_CAPTURE_IDS[:100])

    elif args.dataset == 'MPIIFaceGaze':
        h5_file = GENERATED_DATASET_DIR / 'MPIIFaceGaze_entire.h5'
        dataset = dataset_generator(h5_file, participants=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14])

    elif args.dataset == 'EyeDiap':
        h5_file = GENERATED_DATASET_DIR / 'EyeDiap_entire.h5'
        dataset = dataset_generator(h5_file, participants=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15])

    # Visualize the output space
    visualize_output_space(dataset)