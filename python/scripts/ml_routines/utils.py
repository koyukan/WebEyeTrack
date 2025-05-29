import tensorflow as tf

def embedding_consistency_loss(embeddings, pog_labels):
    """
    Contrastive-style embedding loss without margin.
    Encourages embeddings to reflect spatial relationships in gaze (PoG).
    
    embeddings: (B, D) — latent vectors
    pog_labels: (B, 2) — normalized PoG in [-0.5, 0.5]
    """

    # Pairwise distances
    emb_diffs = tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0)  # (B, B, D)
    emb_distances = tf.norm(emb_diffs, axis=-1)  # (B, B)

    pog_diffs = tf.expand_dims(pog_labels, 1) - tf.expand_dims(pog_labels, 0)
    pog_distances = tf.norm(pog_diffs, axis=-1)  # (B, B)

    # Normalize PoG distances to range [0, 1] (optional but helpful)
    pog_distances /= tf.reduce_max(pog_distances) + 1e-6  # avoid div-by-zero

    # We want embedding distances to match PoG distances
    loss = tf.reduce_mean(tf.square(emb_distances - pog_distances))

    return loss

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

def l2_loss(y_true, y_pred):
    """
    Compute L2 loss between true and predicted values.
    Args:
        y_true: Tensor of true values.
        y_pred: Tensor of predicted values.
    Returns:
        Scalar L2 loss.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))