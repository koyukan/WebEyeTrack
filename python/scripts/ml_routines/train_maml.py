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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

from webeyetrack.constants import GIT_ROOT
from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig, build_full_inference_model
from webeyetrack.vis import plot_pog_errors, plot_2d_dist, matplotlib_to_image
from data import (
    load_total_dataset, 
    load_datasets, 
    get_dataset_metadata,
    get_maml_task_dataset
)
from utils import (
    mae_cm_loss,
    l2_loss
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
SAVED_MODELS_DIR = CWD / 'saved_models'

LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

# Constants
IMG_SIZE = 128

# Loss parameters
L2_COEF = 1
SPREAD_COEF = 1
CONT_COEF = 0
CENTER_COEF = 0

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

def match_spread_loss(preds, gts):
    pred_cov = tf.math.reduce_std(preds, axis=0)
    gt_cov = tf.math.reduce_std(gts, axis=0)
    return tf.reduce_mean(tf.square(pred_cov - gt_cov))

def contrastive_gaze_loss(preds, gts, margin=0.1):
    """
    preds: Tensor of shape (B, 2) — predicted PoGs
    gts:   Tensor of shape (B, 2) — ground truth PoGs

    margin: distance in GT space after which predictions are expected to be separated
    """
    # Compute pairwise distances for GT and predictions
    pred_diff = tf.expand_dims(preds, 1) - tf.expand_dims(preds, 0)  # (B, B, 2)
    pred_dists = tf.norm(pred_diff, axis=-1)  # (B, B)

    gt_diff = tf.expand_dims(gts, 1) - tf.expand_dims(gts, 0)  # (B, B, 2)
    gt_dists = tf.norm(gt_diff, axis=-1)  # (B, B)

    # Positive (similar) pairs — encourage proximity
    pos_mask = tf.cast(gt_dists < margin, tf.float32)
    pos_loss = tf.reduce_sum(pos_mask * tf.square(pred_dists))

    # Negative (dissimilar) pairs — push apart
    neg_mask = tf.cast(gt_dists >= margin, tf.float32)
    neg_loss = tf.reduce_sum(neg_mask * tf.square(tf.maximum(0.0, margin - pred_dists)))

    total_pairs = tf.cast(tf.size(gt_dists), tf.float32)
    return (pos_loss + neg_loss) / total_pairs

def center_penalty_loss(preds, center=tf.constant([0.5, 0.5]), strength=0.05):
    dists_to_center = tf.norm(preds - center, axis=-1)  # (B,)
    penalty = tf.maximum(0.0, strength - dists_to_center)  # Penalize predictions inside a small radius
    return tf.reduce_mean(penalty)

def inner_loop(gaze_head, support_x, support_y, inner_lr, model_config):
    features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
    input_list = [support_x[feature] for feature in features]
    with tf.GradientTape() as tape:
        preds = gaze_head(input_list, training=True)

        # DEBUGGING: Plotting the errors
        # fig = plot_pog_errors(
        #     support_y[:, 0], support_y[:, 1],
        #     preds[:, 0], preds[:, 1],
        # )
        # print(support_y - preds)
        # print(tf.abs(support_y - preds).numpy())
        # import pdb; pdb.set_trace()
        # plt.show()
        # l1_loss = mae_cm_loss(support_y, preds, support_x['screen_info'])
        # Compute the l1 loss directly in norm space
        # l1_loss = tf.reduce_mean(tf.norm(support_y - preds, axis=-1))

        gaze_l2_loss = l2_loss(support_y, preds)
        spread_loss = match_spread_loss(preds, support_y)
        cont_loss = contrastive_gaze_loss(preds, support_y)
        center_loss = center_penalty_loss(preds)
        loss = L2_COEF * gaze_l2_loss + CONT_COEF * cont_loss + CENTER_COEF * center_loss + SPREAD_COEF * spread_loss

        metric_gaze_cm = mae_cm_loss(support_y, preds, support_x['screen_info'])

    grads = tape.gradient(loss, gaze_head.trainable_weights)
    adapted_weights = [w - inner_lr * g for w, g in zip(gaze_head.trainable_weights, grads)]

    losses = {
        'gaze_l1_loss': gaze_l2_loss,
        'spread_loss': spread_loss,
        'cont_loss': cont_loss,
        'center_loss': center_loss,
        'loss': loss
    }
    metrics = {
        'gaze_cm_mae': metric_gaze_cm
    }

    return adapted_weights, losses, metrics

def maml_train(
    RUN_DIR,
    model_config,
    encoder_model,
    gaze_mlp,
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
    valid_model = tf.keras.models.clone_model(gaze_mlp)

    # Track best validation loss
    best_val_query_loss = float("inf")
    best_model_path = RUN_DIR / 'maml_gaze_mlp_best.keras'
    best_full_model_path = RUN_DIR / 'full_model_best.keras'

    for step in tqdm(range(steps_outer), desc="Meta Training Steps"):

        # --- Train Task ---
        support_x, support_y, query_x, query_y = next(train_task_iter)
        support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
        query_x['encoder_features'] = encoder_model(query_x['image'], training=False)

        task_model = tf.keras.models.clone_model(gaze_mlp)
        task_model.build(support_x['encoder_features'].shape)
        task_model.set_weights(gaze_mlp.get_weights())

        for _ in range(steps_inner):
            adapted_weights, support_losses, support_metrics = inner_loop(
                task_model, support_x, support_y, inner_lr, model_config
            )
            task_model.set_weights(adapted_weights)

        with tf.GradientTape() as outer_tape:
            features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
            input_list = [query_x[feature] for feature in features]
            query_preds = task_model(input_list, training=True)
            # query_l1_loss = mae_cm_loss(query_y, query_preds, query_x['screen_info'])
            query_l2_loss = l2_loss(query_y, query_preds)
            query_cont_loss = contrastive_gaze_loss(query_preds, query_y)
            query_center_loss = center_penalty_loss(query_preds)
            query_loss = L2_COEF * query_l2_loss + CONT_COEF * query_cont_loss + CENTER_COEF * query_center_loss

            metrics_gaze_cm_mae = mae_cm_loss(query_y, query_preds, query_x['screen_info'])

            query_losses = {
                'gaze_l2_loss': query_l2_loss,
                'cont_loss': query_cont_loss,
                'center_loss': query_center_loss,
                'total_loss': query_loss
            }

        grads = outer_tape.gradient(query_loss, task_model.trainable_weights)
        meta_optimizer.apply_gradients(zip(grads, gaze_mlp.trainable_weights))

        # --- Logging ---
        # if (step + 1) % (steps_outer // 15) == 0:
        if (step + 1) % 5 == 0:
            if tb_writer:
                # Scalars for TensorBoard
                for key, value in support_losses.items():
                    tf.summary.scalar(f'train/support_{key}', value, step=step)
                for key, value in query_losses.items():
                    tf.summary.scalar(f'train/query_{key}', value, step=step)
                for key, value in support_metrics.items():
                    tf.summary.scalar(f'train/support_{key}', value, step=step)
                tf.summary.scalar('train/query_gaze_cm_mae', metrics_gaze_cm_mae, step=step)

                gaze_predictions_fig = plot_pog_errors(
                    query_y.numpy()[:, 0],
                    query_y.numpy()[:, 1],
                    query_preds.numpy()[:, 0],
                    query_preds.numpy()[:, 1],
                )

                # Convert Matplotlib figure to image tensor
                gaze_predictions_image = matplotlib_to_image(gaze_predictions_fig)
                tf.summary.image(f'train/gaze_predictions', np.expand_dims(gaze_predictions_image, axis=0), step=step)

                tb_writer.flush()
            # print(f"Step {step}: Support Loss: {support_loss.numpy():.3f}, Query Loss: {query_loss.numpy():.3f}")

            # --- Validation ---
            # val_losses = []
            val_losses = defaultdict(list)
            for j in tqdm(range(len(val_ids)), desc="Validation Steps"):
                valid_model.set_weights(gaze_mlp.get_weights())
                support_x, support_y, query_x, query_y = next(valid_task_iter)
                support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
                adapted_weights, support_losses, support_metrics = inner_loop(
                    valid_model, support_x, support_y, inner_lr, model_config
                )
                valid_model.set_weights(adapted_weights)
                query_x['encoder_features'] = encoder_model(query_x['image'], training=False)
                features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
                input_list = [query_x[feature] for feature in features]
                query_preds = valid_model(input_list, training=False)
                query_l2_loss = l2_loss(query_y, query_preds)
                query_cont_loss = contrastive_gaze_loss(query_preds, query_y)
                query_center_loss = center_penalty_loss(query_preds)
                query_loss = L2_COEF * query_l2_loss + CONT_COEF * query_cont_loss + CENTER_COEF * query_center_loss
                
                gaze_cm_mae = mae_cm_loss(query_y, query_preds, query_x['screen_info'])

                val_losses['gaze_l2_loss'].append(query_l2_loss.numpy())
                val_losses['cont_loss'].append(query_cont_loss.numpy())
                val_losses['center_loss'].append(query_center_loss.numpy())
                val_losses['total_loss'].append(query_loss.numpy())
                val_losses['gaze_cm_mae'].append(gaze_cm_mae.numpy())

            avg_val_query_loss = np.mean(val_losses['total_loss'])
            print(f"Validation Step {step}: Avg Query Loss = {avg_val_query_loss:.4f}")

            # Save best model
            if avg_val_query_loss < best_val_query_loss:
                best_val_query_loss = avg_val_query_loss
                gaze_mlp.save(best_model_path)

                # Also save thAdame entire model for inference use later
                full_model = build_full_inference_model(encoder_model, gaze_mlp, model_config)
                full_model.save(best_full_model_path)

                print(f"New best model saved at step {step} with val query loss {avg_val_query_loss:.4f}")

            # Log validation loss
            if tb_writer:
                # tf.summary.scalar('valid_query_loss', avg_val_query_loss, step=step)
                for key, value in val_losses.items():
                    tf.summary.scalar(f'val/{key}', np.mean(value), step=step)

                gaze_predictions_fig = plot_pog_errors(
                    query_y.numpy()[:, 0],
                    query_y.numpy()[:, 1],
                    query_preds.numpy()[:, 0],
                    query_preds.numpy()[:, 1],
                )

                # Convert Matplotlib figure to image tensor
                gaze_predictions_image = matplotlib_to_image(gaze_predictions_fig)
                tf.summary.image(f'val/gaze_predictions', np.expand_dims(gaze_predictions_image, axis=0), step=step)
                
                tb_writer.flush()

    # Save final model
    # gaze_mlp.save_weights(RUN_DIR / 'maml_gaze_mlp_final.h5')
    print("MAML Training completed.")

    # Return the best model
    gaze_mlp.load_weights(best_model_path)
    print(f"Best model loaded from {best_model_path}")
    return gaze_mlp

def maml_test(
    RUN_DIR,
    model_config,
    encoder_model,
    gaze_mlp,
    test_maml_dataset,
    ids,
    inner_lr=0.01,
    steps_inner=5,
    tb_writer=None,
    steps_test=None, # Optional cap on number of test tasks to evaluate
    plots=False
):

    print("Starting MAML Test...")

    test_task_iter = iter(test_maml_dataset)
    train_ids, val_ids, test_ids = ids
    max_steps = steps_test or len(test_ids)

    all_query_losses = []
    all_query_l2_losses = []
    all_query_mae_cm = []

    query_gt_pog_norms = []
    query_pred_pog_norms = []
    support_gt_pog_norms = []
    support_pred_pog_norms = []

    for step in tqdm(range(max_steps), desc="Meta Test Steps"):
        # Sample test task
        support_x, support_y, query_x, query_y = next(test_task_iter)

        # Encode features (encoder is frozen)
        support_x['encoder_features'] = encoder_model(support_x['image'], training=False)
        query_x['encoder_features'] = encoder_model(query_x['image'], training=False)

        # Clone the gaze model (meta-initialized)
        task_model = tf.keras.models.clone_model(gaze_mlp)
        task_model.build(support_x['encoder_features'].shape)
        task_model.set_weights(gaze_mlp.get_weights())

        features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
        input_list = [support_x[feature] for feature in features]

        # Adapt on support set
        for _ in range(steps_inner):
            with tf.GradientTape() as tape:
                support_preds = task_model(input_list, training=True)
                support_l2_loss = l2_loss(support_y, support_preds)
                support_cont_loss = contrastive_gaze_loss(support_preds, support_y)
                support_center_loss = center_penalty_loss(support_preds)
                support_loss = L2_COEF * support_l2_loss + CONT_COEF * support_cont_loss + CENTER_COEF * support_center_loss
            grads = tape.gradient(support_loss, task_model.trainable_weights)
            for w, g in zip(task_model.trainable_weights, grads):
                w.assign_sub(inner_lr * g)

        # Store support predictions and ground truth
        support_pred_pog_norms.append(support_preds.numpy())
        support_gt_pog_norms.append(support_y.numpy())

        features = ['encoder_features'] + [x.name for x in model_config.gaze.inputs]
        input_list = [query_x[feature] for feature in features]

        # Evaluate on query set
        query_preds = task_model(input_list, training=False)
        query_l2_loss = l2_loss(query_y, query_preds)
        query_cont_loss = contrastive_gaze_loss(query_preds, query_y).numpy()
        query_center_loss = center_penalty_loss(query_preds).numpy()
        query_loss = L2_COEF * query_l2_loss + CONT_COEF * query_cont_loss + CENTER_COEF * query_center_loss
        query_pred_pog_norms.append(query_preds.numpy())
        query_gt_pog_norms.append(query_y.numpy())

        metric_gaze_cm_mae = mae_cm_loss(query_y, query_preds, query_x['screen_info'])

        all_query_losses.append(query_loss)
        all_query_l2_losses.append(query_l2_loss)
        all_query_mae_cm.append(metric_gaze_cm_mae)

        if tb_writer:
            with tb_writer.as_default():
                tf.summary.scalar("test_query_loss", query_loss, step=step)
                tb_writer.flush()

    # If generate plots, show the distribution of the pred PoG
    if plots:
        query_pred_pog_norms = np.concatenate(query_pred_pog_norms, axis=0)
        query_gt_pog_norms = np.concatenate(query_gt_pog_norms, axis=0)
        support_pred_pog_norms = np.concatenate(support_pred_pog_norms, axis=0)
        support_gt_pog_norms = np.concatenate(support_gt_pog_norms, axis=0)

        for (kind, gt_pog_norms, pred_pog_norms) in [
            ('query', query_gt_pog_norms, query_pred_pog_norms),
            ('support', support_gt_pog_norms, support_pred_pog_norms)
        ]:

            gt_x = gt_pog_norms[:, 0]
            gt_y = gt_pog_norms[:, 1]
            pred_x = pred_pog_norms[:, 0]
            pred_y = pred_pog_norms[:, 1]

            for version in ['pred', 'gt']:

                if version == 'pred':
                    x_vals, y_vals = pred_x, pred_y
                elif version == 'gt':
                    x_vals, y_vals = gt_x, gt_y
                else:
                    raise ValueError("Version must be 'pred' or 'gt'.")

                # Save figure
                fig = plot_2d_dist(x_vals, y_vals, title=f"{version} - 2D Heatmap of Gaze Targets")
                fig.savefig(RUN_DIR / f"maml_test_{kind}_{version}_heatmap.png")
                plt.close(fig)

            # Save figure
            fig = plot_pog_errors(gt_x, gt_y, pred_x, pred_y)
            fig.savefig(RUN_DIR / f"maml_test_{kind}_error_vectors.png")
            plt.close(fig)

    print(f"MAML Test Finished — Avg Query Loss: {np.mean(all_query_losses):.4f}, Avg Query L2 Loss: {np.mean(all_query_l2_losses):.4f}, Avg Query MAE: {np.mean(all_query_mae_cm):.4f}")
    return all_query_losses

def train(args, config):

    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RUN_DIR = LOG_PATH / f"{TIMESTAMP}_{args.exp}"
    os.makedirs(RUN_DIR, exist_ok=True)

    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Modify the weights_fp
    if config['model']['weights_fp'] is not None:
        dir = SAVED_MODELS_DIR / config['model']['weights_fp']
        model_path = dir / 'best_model.keras'
        config['model']['weights_fp'] = model_path
    else:
        print("WARNING: No model weights path provided, using default.")

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
    model.gaze_mlp = maml_train(
        RUN_DIR=RUN_DIR,
        model_config=model_config,
        encoder_model=encoder,
        gaze_mlp=model.gaze_mlp,
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
        RUN_DIR=RUN_DIR,
        model_config=model_config,
        encoder_model=encoder,
        gaze_mlp=model.gaze_mlp,
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
    parser.add_argument("--exp", type=str, required=True, help="Experiment name for logging purposes")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Only config that is for 'maml' type is allowed
    if config['config']['type'] != 'maml':
        raise ValueError("Only 'maml' type configuration is allowed.")

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
    
    train(args, config)
