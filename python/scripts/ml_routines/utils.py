import tensorflow as tf

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