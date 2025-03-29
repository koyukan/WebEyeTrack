import os
import pathlib
import datetime
import json
import argparse
import time
import json
from tqdm import tqdm

import cattrs
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from webeyetrack.constants import GIT_ROOT
from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig
from webeyetrack.tf_utils import (
    angular_loss, 
    angular_distance, 
    parse_tfrecord_fn, 
    GazeVisualizationCallback,
    ImageVisCallback
)
from data import load_total_dataset

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
IMG_SIZE = 128

def load_datasets(h5_file, train_ids, val_ids, test_ids, config):
    
    # Prepare datasets
    print(f"Loading datasets from {h5_file}")
    train_dataset, train_dataset_size = load_total_dataset(h5_file, participants=train_ids, config=config)
    val_dataset, val_dataset_size = load_total_dataset(h5_file, participants=val_ids, config=config)
    test_dataset, test_dataset_size = load_total_dataset(h5_file, participants=test_ids, config=config)
    print(f"Train dataset size: {train_dataset_size}, Validation dataset size: {val_dataset_size}, Test dataset size: {test_dataset_size}")

    # Sanity check
    for img, label in train_dataset.take(1):
        print("Image batch shape:", img.shape)
        print("Label batch shape:", label.shape)
        print("Image min/max:", tf.reduce_min(img).numpy(), tf.reduce_max(img).numpy())
        # print("Label values:", label.numpy())

    return train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size

def train(config):

    LOG_PATH = FILE_DIR / 'logs'
    os.makedirs(LOG_PATH, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_DIR = LOG_PATH / TIMESTAMP
    os.makedirs(RUN_DIR, exist_ok=True)

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    if config['dataset']['name'] == 'MPIIFaceGaze':
        h5_file = GENERATED_DATASET_DIR / 'MPIIFaceGaze_entire.h5'
        train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        val_ids = [13, 14]
        test_ids = []
    elif config['dataset']['name'] == 'GazeCapture':
        h5_file = GENERATED_DATASET_DIR / 'GazeCapture_entire.h5'
        # Load the GazeCapture participant IDs
        with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
            GAZE_CAPTURE_IDS = json.load(f)

        # For testing, only using 10 participants
        if config['dataset']['gazecapture']['num_of_ids'] > 0:
            GAZE_CAPTURE_IDS = GAZE_CAPTURE_IDS[:config['dataset']['gazecapture']['num_of_ids']]
        else:
            GAZE_CAPTURE_IDS = GAZE_CAPTURE_IDS
        
        # Split the GAZE_CAPTURE_IDS into train, validation, and test sets (80-10-10)
        np.random.seed(config['dataset']['seed'])
        np.random.shuffle(GAZE_CAPTURE_IDS)
        num_participants = len(GAZE_CAPTURE_IDS)
        x, y = config['dataset']['train_val_test_split']
        train_size = int(num_participants * x)
        val_size = int(num_participants * y)
        train_ids = GAZE_CAPTURE_IDS[:train_size]
        val_ids = GAZE_CAPTURE_IDS[train_size:train_size+val_size]
        test_ids = GAZE_CAPTURE_IDS[train_size+val_size:]

    train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size = load_datasets(
        h5_file, 
        train_ids, 
        val_ids, 
        test_ids, 
        config
    )

    # Make a learning rate schedule based on the epoch instead of steps
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['optimizer']['learning_rate'],
        decay_steps=train_dataset_size,
        decay_rate=config['optimizer']['decay_rate']
    )

    # Load model
    model_config = cattrs.structure(config['model'], BlazeGazeConfig)
    model = BlazeGaze(model_config)
    model.model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=config['loss']['functions'],
        metrics=config['loss']['metrics']
    )

    # Callbacks
    callbacks = []
    for callback_name in config['callbacks']:
        if callback_name == 'TensorBoard':
            callback = TensorBoard(
                log_dir=RUN_DIR,
                update_freq='epoch', 
                # histogram_freq=1, 
                # profile_batch='5,10'
            )
            callbacks.append(callback)
        elif callback_name == 'ModelCheckpoint':
            callback = ModelCheckpoint(
                filepath=RUN_DIR/"blazegaze-{epoch:02d}-{val_loss:.2f}.h5", 
                monitor="epoch_angular_distance",
                ave_best_only=True, 
                save_weights_only=True,
            )
            callbacks.append(callback)
        elif callback_name == 'GazeVisualizationCallback':
            # Subset the dataset for visualization (use a few samples)
            train_vis_dataset = train_dataset.take(1)
            valid_vis_dataset = val_dataset.take(1)
            # log_dir = RUN_DIR / 'visualizations'
            # os.makedirs(log_dir, exist_ok=True)
            train_callback = GazeVisualizationCallback(
                dataset=train_vis_dataset,
                log_dir=RUN_DIR,
                img_size=IMG_SIZE,
                name='Gaze (Training)'
            )
            callbacks.append(train_callback)
            valid_callback = GazeVisualizationCallback(
                dataset=valid_vis_dataset,
                log_dir=RUN_DIR,
                img_size=IMG_SIZE,
                name='Gaze (Validation)'
            )
            callbacks.append(valid_callback)
        elif callback_name == 'ImageVisCallback':
            # log_dir = RUN_DIR / 'images'
            # os.makedirs(log_dir, exist_ok=True)
            train_vis_dataset = train_dataset.take(1)
            valid_vis_dataset = val_dataset.take(1)
            train_callback = ImageVisCallback(
                dataset=train_vis_dataset,
                log_dir=RUN_DIR,
                img_size=IMG_SIZE,
                name='Image (Training)'
            )
            callbacks.append(train_callback)
            valid_callback = ImageVisCallback(
                dataset=valid_vis_dataset,
                log_dir=RUN_DIR,
                img_size=IMG_SIZE,
                name='Image (Validation)'
            )
            callbacks.append(valid_callback)
        else:
            raise ValueError(f"Invalid callback name: {callback_name}")

    print(f"Callbacks: {callbacks}")

    steps_per_epoch = train_dataset_size // config['training']['batch_size']
    validation_steps = val_dataset_size // config['training']['batch_size']

    # Train model
    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['training']['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Evaluate model
    # results = model.tf_model.evaluate(test_dataset)
    # print(results)
    # print(f"Test Loss: {results[0]}, Test Angular Error (Degrees): {results[1]}")

if __name__ == "__main__":
    # Add arguments to specify the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    
    train(config)