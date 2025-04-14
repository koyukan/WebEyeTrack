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
from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig, build_full_inference_model
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

LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_DIR = LOG_PATH / TIMESTAMP
os.makedirs(RUN_DIR, exist_ok=True)

# Constants
IMG_SIZE = 128

"""
# References
[1] https://github.com/hereismari/tensorflow-maml/blob/master/maml.ipynb
[2] https://www.digitalocean.com/community/tutorials/first-order-maml-algorithm-in-meta-learning
[3] https://github.com/hereismari/tensorflow-maml/blob/master/maml.ipynb
[4] https://gist.github.com/luis-mueller/f23f483c405b0a169bf279f7b02209bc#file-maml-py
[5] https://arxiv.org/pdf/1703.03400.pdf

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

def mae_cm_loss(y_true, y_pred, screen_info):
    """
    Convert normalized predictions and labels to cm using screen_info
    and compute MAE in cm.

    Args:
        y_true: Tensor of shape (batch_size, 2), normalized labels [0,1]
        y_pred: Tensor of shape (batch_size, 2), normalized predictions [0,1]
        screen_info: Tensor of shape (batch_size, 2), in cm: [height, width]

    Returns:
        Scalar MAE loss in cm
    """
    # Convert from normalized [0,1] to cm by multiplying by screen dimensions
    true_cm = y_true * screen_info
    pred_cm = y_pred * screen_info

    return tf.reduce_mean(tf.abs(true_cm - pred_cm))  # MAE in cm

def inner_loop(gaze_head, support_x, support_y, inner_lr, model_config):
    features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
    input_list = [support_x[feature] for feature in features]
    with tf.GradientTape() as tape:
        preds = gaze_head(input_list, training=True)
        loss = mae_cm_loss(support_y, preds, support_x['screen_info'])
    grads = tape.gradient(loss, gaze_head.trainable_weights)
    adapted_weights = [w - inner_lr * g for w, g in zip(gaze_head.trainable_weights, grads)]
    return adapted_weights, loss

def maml_train(
    model_config,
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
    train_task_iter = iter(train_maml_dataset)
    train_ids, val_ids, test_ids = ids

    # Validation model & iterator
    valid_task_iter = iter(valid_maml_dataset)
    valid_model = tf.keras.models.clone_model(gaze_model)

    # Track best validation loss
    best_val_query_loss = float("inf")
    best_model_path = RUN_DIR / 'maml_gaze_model_best.h5'
    best_full_model_path = RUN_DIR / 'full_model_best.h5'

    for step in tqdm(range(steps_outer), desc="Meta Training Steps"):

        # --- Train Task ---
        support_x, support_y, query_x, query_y = next(train_task_iter)
        support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
        query_x['encoder_features'] = encoder_model(query_x['image'], training=False)

        task_model = tf.keras.models.clone_model(gaze_model)
        task_model.build(support_x['encoder_features'].shape)
        task_model.set_weights(gaze_model.get_weights())

        for _ in range(steps_inner):
            adapted_weights, support_loss = inner_loop(
                task_model, support_x, support_y, inner_lr, model_config
            )
            task_model.set_weights(adapted_weights)

        with tf.GradientTape() as outer_tape:
            features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
            input_list = [query_x[feature] for feature in features]
            query_preds = task_model(input_list, training=True)
            query_loss = mae_cm_loss(query_y, query_preds, query_x['screen_info'])

        grads = outer_tape.gradient(query_loss, task_model.trainable_weights)
        meta_optimizer.apply_gradients(zip(grads, gaze_model.trainable_weights))

        # --- Logging ---
        if (step + 1) % (steps_outer // 15) == 0:
            if tb_writer:
                tf.summary.scalar('support_loss', support_loss, step=step)
                tf.summary.scalar('query_loss', query_loss, step=step)
                tb_writer.flush()
            print(f"Step {step}: Support Loss: {support_loss.numpy():.3f}, Query Loss: {query_loss.numpy():.3f}")

            # --- Validation ---
            val_losses = []
            for j in tqdm(range(len(val_ids)), desc="Validation Steps"):
                valid_model.set_weights(gaze_model.get_weights())
                support_x, support_y, query_x, query_y = next(valid_task_iter)
                support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
                adapted_weights, support_loss = inner_loop(
                    valid_model, support_x, support_y, inner_lr, model_config
                )
                valid_model.set_weights(adapted_weights)
                query_x['encoder_features'] = encoder_model(query_x['image'], training=False)
                features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
                input_list = [query_x[feature] for feature in features]
                query_loss = mae_cm_loss(query_y, valid_model(input_list, training=False), query_x['screen_info'])
                val_losses.append(query_loss.numpy())

            avg_val_query_loss = np.mean(val_losses)
            print(f"Validation Step {step}: Avg Query Loss = {avg_val_query_loss:.4f}")

            # Save best model
            if avg_val_query_loss < best_val_query_loss:
                best_val_query_loss = avg_val_query_loss
                gaze_model.save_weights(best_model_path)

                # Also save the entire model for inference use later
                full_model = build_full_inference_model(encoder_model, gaze_model, model_config)
                full_model.save(best_full_model_path)

                print(f"New best model saved at step {step} with val query loss {avg_val_query_loss:.4f}")

            # Log validation loss
            if tb_writer:
                tf.summary.scalar('valid_query_loss', avg_val_query_loss, step=step)
                tb_writer.flush()

    # Save final model
    # gaze_model.save_weights(RUN_DIR / 'maml_gaze_model_final.h5')
    print("MAML Training completed.")

    # Return the best model
    gaze_model.load_weights(best_model_path)
    print(f"Best model loaded from {best_model_path}")
    return gaze_model

def maml_test(
    model_config,
    encoder_model,
    gaze_model,
    test_maml_dataset,
    ids,
    inner_lr=0.01,
    steps_inner=1,
    tb_writer=None,
    steps_test=None  # Optional cap on number of test tasks to evaluate
):

    print("Starting MAML Test...")

    test_task_iter = iter(test_maml_dataset)
    train_ids, val_ids, test_ids = ids
    max_steps = steps_test or len(test_ids)

    all_query_losses = []

    for step in tqdm(range(max_steps), desc="Meta Test Steps"):
        # Sample test task
        support_x, support_y, query_x, query_y = next(test_task_iter)

        # Encode features (encoder is frozen)
        support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
        query_x['encoder_features'] = encoder_model(query_x['image'], training=False)

        # Clone the gaze model (meta-initialized)
        task_model = tf.keras.models.clone_model(gaze_model)
        task_model.build(support_x['encoder_features'].shape)
        task_model.set_weights(gaze_model.get_weights())

        features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
        input_list = [support_x[feature] for feature in features]

        # Adapt on support set
        for _ in range(steps_inner):
            with tf.GradientTape() as tape:
                support_preds = task_model(input_list, training=True)
                support_loss = mae_cm_loss(support_y, support_preds, support_x['screen_info'])
            grads = tape.gradient(support_loss, task_model.trainable_weights)
            for w, g in zip(task_model.trainable_weights, grads):
                w.assign_sub(inner_lr * g)

        features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
        input_list = [query_x[feature] for feature in features]

        # Evaluate on query set
        query_preds = task_model(input_list, training=False)
        query_loss = mae_cm_loss(query_y, query_preds, query_x['screen_info']).numpy()

        all_query_losses.append(query_loss)

        if tb_writer:
            with tb_writer.as_default():
                tf.summary.scalar("test_query_loss", query_loss, step=step)
                tb_writer.flush()

    print(f"MAML Test Finished — Avg Query Loss (MAE): {np.mean(all_query_losses):.4f}")
    return all_query_losses

def train(config):

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Modify the encoder_weights_fp
    if config['model']['encoder']['weights_fp'] is not None:
        dir = SAVED_MODELS_DIR / config['model']['encoder']['weights_fp']

        # Find the .h5 file that starts with 'encoder'
        encoder_weights_fp = None
        for file in dir.glob('*.h5'):
            if file.name.startswith('encoder'):
                encoder_weights_fp = file
                break
        if encoder_weights_fp is None:
            raise ValueError(f"No encoder weights file found in {dir}")
        config['model']['encoder']['weights_fp'] = encoder_weights_fp
        print(f"Encoder weights path: {config['model']['encoder']['weights_fp']}")
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
    test_maml_dataset = get_maml_task_dataset(
        h5_file,
        test_ids,
        config
    )
    
    # Running the MAML training
    model.gaze_model = maml_train(
        model_config=model_config,
        encoder_model=encoder,
        gaze_model=model.gaze_model,
        train_maml_dataset=train_maml_dataset,
        valid_maml_dataset=valid_maml_dataset,
        ids=(train_ids, val_ids, test_ids),
        inner_lr=config['optimizer']['inner_lr'],
        outer_lr=config['optimizer']['outer_lr'],
        steps_outer=config['training']['num_outer_steps'],
        steps_inner=config['training']['num_inner_steps'],
        tb_writer=tb_writer
    )

    # Run the test dataset
    maml_test(
        model_config=model_config,
        encoder_model=encoder,
        gaze_model=model.gaze_model,
        test_maml_dataset=test_maml_dataset,
        ids=(train_ids, val_ids, test_ids),
        inner_lr=config['optimizer']['inner_lr'],
        steps_inner=config['training']['num_inner_steps'],
        tb_writer=tb_writer
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
    
    train(config)
