import tensorflow as tf
import cv2

def apply(img):
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
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation, random_gaussian_noise, random_blur]

    # Geometric augmentations
    geometric_methods = []

    # Apply augmentations
    for augmentation_method in geometric_methods + color_methods:
        img = randomly_apply_operation(augmentation_method, img)

    # Ensure image pixel values are within valid range
    img = tf.clip_by_value(img, 0., 1.)
    return img


def get_random_bool():
    """Generate a random boolean."""
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(operation, img, *args):
    """Randomly apply the given augmentation method."""
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, *args),
        lambda: (img)
    )


### Color Augmentations (Image Only)
def random_brightness(img, max_delta=0.12):
    """Randomly change brightness."""
    return tf.image.random_brightness(img, max_delta)


def random_contrast(img, lower=0.5, upper=1.5):
    """Randomly change contrast."""
    return tf.image.random_contrast(img, lower, upper)


def random_hue(img, max_delta=0.08):
    """Randomly change hue."""
    return tf.image.random_hue(img, max_delta)


def random_saturation(img, lower=0.5, upper=1.5):
    """Randomly change saturation."""
    return tf.image.random_saturation(img, lower, upper)


def random_gaussian_noise(img, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=stddev, dtype=tf.float32)
    return tf.clip_by_value(img + noise, 0., 1.)


def random_grayscale(img):
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
    return img


def random_blur(img, ksize=5):
    img = tf.numpy_function(lambda x: cv2.GaussianBlur(x, (ksize, ksize), 0), [img], tf.float32)
    return img

