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
    load_datasets
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
print(f"Run directory: {RUN_DIR}")

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

    # Get dataset metadata
    h5_file, train_ids, val_ids, test_ids = get_dataset_metadata(config)

    # Load the per-person dataset
    train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size = load_datasets(
        h5_file, 
        train_ids, 
        val_ids, 
        test_ids, 
        config
    )

    # Run the test dataset through the autoencoder (only the test dataset)
    for step, (image, image) in tqdm(enumerate(test_dataset), total=test_dataset_size // config['training']['batch_size']):
        # Get the encoder output
        encoder_output = encoder(image, training=False)
        print(encoder_output.numpy())
        import pdb; pdb.set_trace()
        break


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

    # Ask confirmation for evaluation with a 5 second loading bar
    print("Starting evaluation in 5 seconds...\n")
    for i in tqdm(range(5)):    
        time.sleep(1)

    # Start evaluation
    print("Evaluating...")
    
    eval(config)