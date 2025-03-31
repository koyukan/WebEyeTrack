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
from data import (
    load_total_dataset, 
    load_datasets, 
    get_dataset_metadata,
    get_maml_task_dataset
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
SAVED_MODELS_DIR = CWD / 'saved_models'

# Constants
IMG_SIZE = 128

"""
Initialize meta-model θ (e.g., encoder + head)

For each meta-epoch:
    Sample a batch of tasks T_i ~ p(T)        # Each task = 1 participant

    meta_grads = 0

    For each task T_i:
        # Split into support and query sets
        (x_supp, y_supp), (x_query, y_query) = sample_task(T_i)

        # --- Inner Loop ---
        Clone the model θ_i ← θ

        For k steps:
            Compute support loss: L_supp = loss(y_supp, θ_i(x_supp))
            Compute gradients ∇L_supp w.r.t. θ_i
            Update θ_i ← θ_i - α * ∇L_supp     # α = inner learning rate

        # --- Outer Loop ---
        Compute query loss: L_query = loss(y_query, θ_i(x_query))
        Compute gradients ∇L_query w.r.t. θ   # NOT θ_i

        Accumulate ∇L_query into meta_grads

    Average meta_grads over tasks
    Update meta-model: θ ← θ - β * meta_grads     # β = outer learning rate
"""

def inner_loop(gaze_head, support_features, support_y, inner_lr, loss_fn):
    with tf.GradientTape() as tape:
        preds = gaze_head(support_features, training=True)
        loss = loss_fn(support_y, preds)
    grads = tape.gradient(loss, gaze_head.trainable_weights)
    adapted_weights = [w - inner_lr * g for w, g in zip(gaze_head.trainable_weights, grads)]
    return adapted_weights, loss

def maml_train(
    encoder_model,
    gaze_model,
    train_maml_dataset,
    valid_maml_dataset,
    ids,
    steps_outer=1000,
    inner_lr=0.01,
    outer_lr=0.001,
    steps_inner=1,
    tb_writer=None
):
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_task_iter = iter(train_maml_dataset)
    train_ids, val_ids, test_ids = ids

    # Model and optimizer for validation
    valid_task_iter = iter(valid_maml_dataset)
    valid_model = tf.keras.models.clone_model(gaze_model)
    valid_optim = tf.keras.optimizers.Adam(learning_rate=inner_lr)
    valid_tasks = 100

    for step in tqdm(range(steps_outer), desc="Meta Training Steps"):

        # Sample a task
        support_x, support_y, query_x, query_y = next(train_task_iter)

        # Step 1: Extract features
        support_features = encoder_model(support_x, training=False)
        query_features = encoder_model(query_x, training=False)

        # Step 2: Clone gaze head
        task_model = tf.keras.models.clone_model(gaze_model)
        task_model.build(support_features.shape)  # Ensure weights exist
        task_model.set_weights(gaze_model.get_weights())

        # Step 3: Inner loop → get adapted weights (don't assign them!)
        for i in range(steps_inner):
            adapted_weights, support_loss = inner_loop(
                task_model,
                support_features,
                support_y,
                inner_lr,
                loss_fn
            )
            task_model.set_weights(adapted_weights)

        # Step 4: Outer loop
        with tf.GradientTape() as outer_tape:
            query_preds = task_model(query_features, training=True)
            query_loss = loss_fn(query_y, query_preds)

        grads = outer_tape.gradient(query_loss, task_model.trainable_weights) # Full grads
        meta_optimizer.apply_gradients(zip(grads, gaze_model.trainable_weights))

        # Report the loss to TensorBoard
        if (step + 1) % 10 == 0:
            if tb_writer:
                tf.summary.scalar('support_loss', support_loss, step=step)
                tf.summary.scalar('query_loss', query_loss, step=step)
                tb_writer.flush()
            print(f"Step {step}: Support Loss: {support_loss.numpy():.3f}, Query Loss: {query_loss.numpy():.3f}")

            # Perform validation
            losses = []
            for j in tqdm(range(len(val_ids)), desc="Validation Steps"):
                valid_model.set_weights(gaze_model.get_weights())
                support_x, support_y, query_x, query_y = next(valid_task_iter)
                support_features = encoder_model(support_x, training=False)
                adapted_weights, support_loss = inner_loop(
                    valid_model,
                    support_features,
                    support_y,
                    inner_lr,
                    loss_fn
                )
                valid_model.set_weights(adapted_weights)
                query_features = encoder_model(query_x, training=False)
                query_loss = loss_fn(query_y, valid_model(query_features, training=False))
                losses.append((query_loss.numpy(), support_loss.numpy()))
            querry_losses, support_losses = zip(*losses)
            if tb_writer:
                tf.summary.scalar('valid_support_loss', np.mean(support_losses), step=step)
                tf.summary.scalar('valid_query_loss', np.mean(querry_losses), step=step)
                tb_writer.flush()
            print(f"Validation Step {step}: Support Loss: {np.mean(support_losses):.3f}, Query Loss: {np.mean(querry_losses):.3f}")

    # Save the MAML parameters
    ...

def train(config):

    LOG_PATH = FILE_DIR / 'logs'
    os.makedirs(LOG_PATH, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_DIR = LOG_PATH / TIMESTAMP
    os.makedirs(RUN_DIR, exist_ok=True)

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Modify the encoder_weights_fp
    if config['model']['encoder_weights_fp'] is not None:
        dir = SAVED_MODELS_DIR / config['model']['encoder_weights_fp']

        # Find the .h5 file that starts with 'encoder'
        encoder_weights_fp = None
        for file in dir.glob('*.h5'):
            if file.name.startswith('encoder'):
                encoder_weights_fp = file
                break
        if encoder_weights_fp is None:
            raise ValueError(f"No encoder weights file found in {dir}")
        config['model']['encoder_weights_fp'] = encoder_weights_fp
        print(f"Encoder weights path: {config['model']['encoder_weights_fp']}")
    else:
        print("No encoder weights path provided, using default.")

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

    # Load MAML dataset
    train_maml_dataset = get_maml_task_dataset(
        h5_file,
        train_ids,
        config
    )
    valid_maml_dataset = get_maml_task_dataset(
        h5_file,
        val_ids,
        config
    )
    
    # Running the MAML training
    maml_train(
        encoder_model=encoder,
        gaze_model=model.gaze_model,
        train_maml_dataset=train_maml_dataset,
        valid_maml_dataset=valid_maml_dataset,
        ids=(train_ids, val_ids, test_ids),
        inner_lr=config['optimizer']['inner_lr'],
        outer_lr=config['optimizer']['learning_rate'],
        steps_outer=config['training']['num_outer_steps'],
        steps_inner=config['training']['num_inner_steps'],
        tb_writer=tb_writer
    )
    print("Training completed.")

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
    
    train(config)
