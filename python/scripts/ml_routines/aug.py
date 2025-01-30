import tensorflow as tf

def apply(img, gaze_vector):
    """
    Randomly applying data augmentation methods to the image and gaze vector.
    
    Args:
        img: Tensor of shape (height, width, depth).
        gaze_vector: Tensor of shape (3,), representing the gaze direction.

    Returns:
        augmented_img: Tensor of shape (height, width, depth).
        augmented_gaze_vector: Tensor of shape (3,), representing the augmented gaze direction.
    """
    # Color augmentations
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]

    # Geometric augmentations
    geometric_methods = []

    # Apply augmentations
    for augmentation_method in geometric_methods + color_methods:
        img, gaze_vector = randomly_apply_operation(augmentation_method, img, gaze_vector)

    # Ensure image pixel values are within valid range
    img = tf.clip_by_value(img, 0., 1.)
    return img, gaze_vector


def get_random_bool():
    """Generate a random boolean."""
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(operation, img, gaze_vector, *args):
    """Randomly apply the given augmentation method."""
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gaze_vector, *args),
        lambda: (img, gaze_vector)
    )


### Color Augmentations (Image Only)
def random_brightness(img, gaze_vector, max_delta=0.12):
    """Randomly change brightness."""
    return tf.image.random_brightness(img, max_delta), gaze_vector


def random_contrast(img, gaze_vector, lower=0.5, upper=1.5):
    """Randomly change contrast."""
    return tf.image.random_contrast(img, lower, upper), gaze_vector


def random_hue(img, gaze_vector, max_delta=0.08):
    """Randomly change hue."""
    return tf.image.random_hue(img, max_delta), gaze_vector


def random_saturation(img, gaze_vector, lower=0.5, upper=1.5):
    """Randomly change saturation."""
    return tf.image.random_saturation(img, lower, upper), gaze_vector
