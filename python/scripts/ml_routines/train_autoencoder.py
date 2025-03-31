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
    ImageVisCallback, 
    EncoderDecoderCheckpoint
)
from data import load_total_dataset, load_datasets, get_dataset_metadata

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
IMG_SIZE = 128

def train(config):

    LOG_PATH = FILE_DIR / 'logs'
    os.makedirs(LOG_PATH, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_DIR = LOG_PATH / TIMESTAMP
    os.makedirs(RUN_DIR, exist_ok=True)

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Get dataset metadata
    h5_file, train_ids, val_ids, test_ids = get_dataset_metadata(config)

    # Load datasets
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

        elif callback_name == 'EncoderDecoderCheckpoint':
            callback = EncoderDecoderCheckpoint(
                encoder=model.encoder,
                decoder=model.decoder,
                checkpoint_dir=RUN_DIR,
                monitor='val_loss',  # or 'val_epoch_angular_distance'
                mode='min'
            )
            callbacks.append(callback)
            
        elif callback_name == 'GazeVisualizationCallback':
            train_vis_dataset = train_dataset.take(1)
            valid_vis_dataset = val_dataset.take(1)
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
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config['config']['type'] != 'autoencoder':
        raise ValueError("Only 'autoencoder' type configuration is allowed.")

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
