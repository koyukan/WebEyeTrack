import os
from typing import Dict
import pathlib

import tensorflow as tf
import tensorflowjs as tfjs

def save_model(models: Dict[str, tf.keras.Model], output_dir: pathlib.Path):
    
    for name, model in models.items():
        model_path = output_dir / f"{name}.h5"
        model.save(model_path, save_format='h5')
        model_js_path = output_dir / f"tfjs_{name}"
        os.makedirs(model_js_path, exist_ok=True)
        tfjs.converters.save_keras_model(model, model_js_path)
        print(f"Model '{name}' saved to {model_path}")

def embedding_consistency_loss(embeddings, pog_labels, sample_weight=None):
    """
    Weighted contrastive-style loss on embedding distances to reflect gaze (PoG) similarity.
    
    embeddings: (B, D) latent vectors
    pog_labels: (B, 2) normalized PoG
    sample_weight: Optional (B,) vector of per-sample weights
    """

    # Pairwise distances
    emb_diffs = tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0)  # (B, B, D)
    emb_distances = tf.norm(emb_diffs, axis=-1)  # (B, B)

    pog_diffs = tf.expand_dims(pog_labels, 1) - tf.expand_dims(pog_labels, 0)  # (B, B, 2)
    pog_distances = tf.norm(pog_diffs, axis=-1)  # (B, B)

    # Normalize distances to [0, 1]
    pog_distances /= tf.reduce_max(pog_distances) + 1e-6

    loss_matrix = tf.square(emb_distances - pog_distances)  # (B, B)

    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        # Broadcast to (B, B) for pairwise use
        weight_matrix = tf.expand_dims(sample_weight, 1) * tf.expand_dims(sample_weight, 0)
        loss_matrix *= weight_matrix

    return tf.reduce_mean(loss_matrix)

def compute_batch_ssim(gt_images: tf.Tensor, recon_images: tf.Tensor, max_val=1.0):
    """
    Compute mean SSIM over a batch of reconstructed vs ground truth images.

    Args:
        gt_images (tf.Tensor): Ground truth images, shape (B, H, W, C)
        recon_images (tf.Tensor): Reconstructed images, shape (B, H, W, C)
        max_val (float): Maximum possible pixel value (e.g., 1.0 if images are normalized)

    Returns:
        tf.Tensor: Scalar average SSIM over batch
    """
    ssim_scores = tf.image.ssim(gt_images, recon_images, max_val=max_val)
    return tf.reduce_mean(ssim_scores)

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

    # return tf.reduce_mean(tf.abs(true_cm - pred_cm))  # MAE in cm
    return tf.reduce_mean(
        tf.norm(true_cm - pred_cm, axis=-1)
    )

def l2_loss(y_true, y_pred, sample_weight=None):
    """
    Compute L2 loss between true and predicted values, optionally weighted.
    
    Args:
        y_true: Tensor of shape (B, 2)
        y_pred: Tensor of shape (B, 2)
        sample_weight: Optional tensor of shape (B,)
    Returns:
        Scalar loss.
    """
    loss = tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)  # (B,)

    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        loss = loss * sample_weight
        return tf.reduce_mean(loss)
    else:
        return tf.reduce_mean(loss)