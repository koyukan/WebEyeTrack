import os
import sys
import pathlib
from collections import defaultdict
import argparse
import json
import time
import datetime

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml
import cattrs
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig
from webeyetrack.constants import GIT_ROOT

CWD = pathlib.Path(__file__).parent
from train_maml import maml_test
from data import (
    get_dataset_metadata,
    get_maml_task_dataset
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
SAVED_MODELS_DIR = CWD / 'saved_models'

OUTPUT_DIR = FILE_DIR / 'logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_DIR = OUTPUT_DIR / TIMESTAMP
os.makedirs(RUN_DIR, exist_ok=True)

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
PER_PARTICIPANT_SIZE = 30

def eval(config):

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create tensorboard writer
    tb_writer = tf.summary.create_file_writer(str(RUN_DIR))
    tb_writer.set_as_default()

    # Load model
    model_config = cattrs.structure(config['model'], BlazeGazeConfig)
    model = BlazeGaze(model_config)
    model.freeze_encoder()

    encoder = model.encoder
    gaze_model = model.gaze_model

    # Get dataset metadata
    h5_file, train_ids, val_ids, test_ids = get_dataset_metadata(config)

    # Load MAML dataset
    # train_maml_dataset = get_maml_task_dataset(
    #     h5_file,
    #     train_ids,
    #     config
    # )
    # valid_maml_dataset = get_maml_task_dataset(
    #     h5_file,
    #     val_ids,
    #     config
    # )
    test_maml_dataset = get_maml_task_dataset(
        h5_file,
        test_ids,
        config
    )

    # Run the test dataset
    maml_test(
        model_config=model_config,
        encoder_model=encoder,
        gaze_model=gaze_model,
        test_maml_dataset=test_maml_dataset,
        ids=(train_ids, val_ids, test_ids),
        inner_lr=config['optimizer']['inner_lr'],
        steps_inner=config['training']['num_inner_steps'],
        tb_writer=tb_writer,
        plots=True
    )

if __name__ == "__main__":
    # Add arguments to specify the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Only config that is for 'gaze' type is allowed
    if config['config']['type'] != 'gaze':
        raise ValueError("Only 'gaze' type configuration is allowed.")

    # Print the configuration
    print("\n")
    print("#" * 80)
    print("Configuration:")
    print(json.dumps(config, indent=4))
    print("#" * 80)
    print("\n")

    # Ask confirmation for training with a 5 second loading bar
    print("Starting training in 5 seconds...\n")
    for i in tqdm(range(5)):    
        time.sleep(1)

    # Start training
    print("Training...")
    
    eval(config)