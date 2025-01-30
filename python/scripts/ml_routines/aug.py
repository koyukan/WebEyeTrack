import tensorflow as tf
import cv2

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
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation, random_gaussian_noise, random_grayscale, random_blur]

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


def random_gaussian_noise(img, gaze_vector, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=stddev, dtype=tf.float32)
    return tf.clip_by_value(img + noise, 0., 1.), gaze_vector


def random_grayscale(img, gaze_vector):
    """
    Converts the image to grayscale while keeping the original (128, 128, 3) shape.
    
    Args:
        img: Tensor of shape (128, 128, 3).
        gaze_vector: Corresponding gaze vector.

    Returns:
        Grayscale image with shape (128, 128, 3).
        Unchanged gaze_vector.
    """
    img = tf.image.rgb_to_grayscale(img)  # Converts to (128, 128, 1)
    img = tf.image.grayscale_to_rgb(img)  # Converts back to (128, 128, 3)
    return img, gaze_vector


def random_blur(img, gaze_vector, ksize=5):
    img = tf.numpy_function(lambda x: cv2.GaussianBlur(x, (ksize, ksize), 0), [img], tf.float32)
    return img, gaze_vector

