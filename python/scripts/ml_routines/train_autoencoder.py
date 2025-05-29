import os
import pathlib
import datetime
import json
import argparse
import time
import json
from collections import defaultdict

from tqdm import tqdm
import cattrs
import yaml
import numpy as np
import tensorflow as tf

from webeyetrack.constants import GIT_ROOT
from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig
import webeyetrack.vis as vis
from data import load_total_dataset, load_datasets, get_dataset_metadata
from utils import (
    mae_cm_loss,
    l2_loss,
    compute_batch_ssim,
    embedding_consistency_loss
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
IMG_SIZE = 128

RECONT_COEF = 1
CONSISTENCY_COEF = 0.1
GAZE_COEF = 1

def train(config):

    LOG_PATH = FILE_DIR / 'logs'
    os.makedirs(LOG_PATH, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_DIR = LOG_PATH / TIMESTAMP
    os.makedirs(RUN_DIR, exist_ok=True)

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create tensorboard writer
    tb_writer = tf.summary.create_file_writer(str(RUN_DIR))
    tb_writer.set_as_default()

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

    steps_per_epoch = train_dataset_size // config['training']['batch_size']
    validation_steps = val_dataset_size // config['training']['batch_size']

    # Train dataset iterator
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_dataset_iter = iter(train_dataset)

    # DEBUG: Get a single sample to check the data pipeline
    # sample = next(train_dataset_iter)

    for epoch in tqdm(range(config['training']['epochs']), desc="Training Epochs"):
        losses = defaultdict(list)
        metrics = defaultdict(list)
        embeddings_to_pog = defaultdict(list)

        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)
        progress_bar.set_postfix({"loss": "N/A"})
        
        for step in range(steps_per_epoch):
            sample = next(train_dataset_iter)
            progress_bar.update(1)

            # Prepare input features
            features = ['image'] + [x.name for x in model_config.gaze.inputs]
            input_list = [sample[feature] for feature in features]

            # Calculate loss
            with tf.GradientTape() as tape:
                preds = model.model(input_list, training=True) # encoder embedding, gaze preds, and decoder output
                gaze_loss = l2_loss(preds[1], sample['pog_norm'])
                decoder_loss = tf.reduce_mean(tf.square(preds[2] - sample['image']))
                consistency_loss = embedding_consistency_loss(
                    preds[0],
                    sample['pog_norm'],
                )
                total_loss = GAZE_COEF * gaze_loss + RECONT_COEF * decoder_loss + CONSISTENCY_COEF * consistency_loss
                
                losses['gaze_l2_loss'].append(gaze_loss.numpy())
                losses['decoder_l1_loss'].append(decoder_loss.numpy())
                losses['embedding_consistency_loss'].append(consistency_loss.numpy())
                losses['loss'].append(total_loss.numpy())

                # Compute metrics
                metrics['ssim'].append(
                    compute_batch_ssim(sample['image'], preds[2])
                )
                metrics['gaze_cm_mae'].append(
                    mae_cm_loss(
                        sample['pog_norm'],
                        preds[1],
                        sample['screen_info']
                    )
                )

                # Store the embeddings and their corresponding PoG labels for visualization
                embeddings_to_pog['embeddings'].append(preds[0].numpy())
                embeddings_to_pog['pog_labels'].append(sample['pog_norm'].numpy())

            # Compute gradients and update model weights
            gradients = tape.gradient(total_loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))
            for key, value in losses.items():
                progress_bar.set_postfix({key: np.mean(value)})

            # if step >= 5:
            #     break

        # Log average loss for the epoch to tensorboard
        if tb_writer:
            # Scalars
            for key, value in losses.items():
                tf.summary.scalar(f'train/{key}', np.mean(value), step=epoch)
            for key, value in metrics.items():
                tf.summary.scalar(f'train/{key}', np.mean(value), step=epoch)

            # Images (reconstruction and gaze predictions)
            image_reconstruction = vis.draw_reconstruction(
                sample['image'].numpy(),
                preds[2].numpy(),
            )
            tf.summary.image('train/image_reconstruction', image_reconstruction, step=epoch)

            gaze_predictions_fig = vis.plot_pog_errors(
                sample['pog_norm'].numpy()[:, 0],
                sample['pog_norm'].numpy()[:, 1],
                preds[1].numpy()[:, 0],
                preds[1].numpy()[:, 1],
            )

            # Convert Matplotlib figure to image tensor
            gaze_predictions_image = vis.matplotlib_to_image(gaze_predictions_fig)
            tf.summary.image('train/gaze_predictions', np.expand_dims(gaze_predictions_image, axis=0), step=epoch)

            # Convert the embeddings and PoG labels to numpy arrays
            embeddings_np = np.concatenate(embeddings_to_pog['embeddings'], axis=0)
            pog_labels_np = np.concatenate(embeddings_to_pog['pog_labels'], axis=0)
            print(f"Embeddings shape: {embeddings_np.shape}, PoG labels shape: {pog_labels_np.shape}")

            # Check the quality of the embeddings
            tsne_embeddings_fig = vis.plot_tsne_colored_by_pog(
                embeddings_np,
                pog_labels_np
            )
            tsne_embeddings_image = vis.matplotlib_to_image(tsne_embeddings_fig)
            tf.summary.image('train/tsne_embeddings', np.expand_dims(tsne_embeddings_image, axis=0), step=epoch)

            # Flush the writer
            tb_writer.flush()

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {np.mean(losses['loss'])}")

if __name__ == "__main__":
    # Add arguments to specify the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate the configuration
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
