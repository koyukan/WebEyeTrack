import pathlib
import json

import tensorflow as tf
import numpy as np
import cv2

from webeyetrack.constants import GIT_ROOT
from webeyetrack.utilities import pitch_yaw_to_gaze_vector

from .aug import apply

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# NORM_FILE_FP = GENERATED_DATASET_DIR / 'MPIIFaceGaze_blazegaze_mean_std.json'
# with open(NORM_FILE_FP, 'r') as f:
#     norm_data = json.load(f)

IMG_SIZE = 128

# Fourier Transform
# https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb

class EncoderDecoderCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, encoder, decoder, checkpoint_dir, monitor='val_loss', mode='min'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')

        # Filenames
        self.encoder_fp = self.checkpoint_dir / "encoder_best.h5"
        self.decoder_fp = self.checkpoint_dir / "decoder_best.h5"

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        is_better = (
            current < self.best if self.mode == 'min' else current > self.best
        )
        if is_better:
            self.best = current
            print(f"\nSaving encoder & decoder weights at epoch {epoch + 1} with {self.monitor} = {current:.4f}")

            # Clean up any older checkpoint files
            for file in self.checkpoint_dir.glob("encoder_best_*.h5"):
                file.unlink()
            for file in self.checkpoint_dir.glob("decoder_best_*.h5"):
                file.unlink()

            # Save with epoch-specific filenames
            encoder_path = self.checkpoint_dir / f"encoder_best_epoch_{epoch+1:02d}.h5"
            decoder_path = self.checkpoint_dir / f"decoder_best_epoch_{epoch+1:02d}.h5"

            self.encoder.save_weights(encoder_path)
            self.decoder.save_weights(decoder_path)

            # Optionally update stored filenames (for potential later cleanup or tracking)
            self.encoder_fp = encoder_path
            self.decoder_fp = decoder_path

class GazeVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir, img_size, name='Gaze'):
        super().__init__()
        self.dataset = dataset  # Dataset to visualize predictions
        self.log_dir = log_dir  # Directory to store TensorBoard logs
        self.img_size = img_size
        self.name = name
        self.file_writer = tf.summary.create_file_writer(str(log_dir))

    def draw_gaze_vector(self, image, gaze_vector, scale=50, color=(255, 0, 0), thickness=2):
        """
        Draw the gaze vector on the image.

        Args:
            image: A (H, W, 3) NumPy array representing the image.
            gaze_vector: A (3,) vector representing the gaze direction.
            scale: Scaling factor for the length of the gaze vector.
        """
        import cv2
        h, w, _ = image.shape
        center = (w // 2, h // 2)  # Center of the image
        endpoint = (
            int(center[0] + gaze_vector[0] * scale),
            int(center[1] - gaze_vector[1] * scale),
        )
        return cv2.arrowedLine(image, center, endpoint, color, thickness=thickness, tipLength=0.3)

    def on_epoch_end(self, epoch, logs=None):
        # Select a batch from the dataset
        # for images, (gaze_vector, embedding) in self.dataset.take(1):
        for images, gaze_vector in self.dataset.take(1):
            pred_gaze_vectors, pred_gaze_z = self.model.predict(images)  # Get model predictions
            images = images.numpy()  # Convert images to NumPy array
            # gaze_pitch_yaw = gaze_pitch_yaw.numpy()  # Ground truth gaze vectors
            gaze_vector = gaze_vector.numpy()  # Ground truth gaze vectors

            # Draw gaze vectors on the images
            visualizations = []
            for i in range(len(images)):

                # Denormalize image
                # images[i] = (images[i] * norm_data['image_std']) + norm_data['image_mean']
                uint8_image = (images[i] * 255).astype(np.uint8)
                # uint8_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)

                # import pdb; pdb.set_trace()
                gt_gaze_vector = gaze_vector[i]
                pred_gaze_vector = pred_gaze_vectors[i]
                # Convert gaze vector to pitch and yaw
                # gt_gaze_pitch_yaw = vector_to_pitch_yaw(gaze_vector[i])

                # Convert pitch and yaw in radians to degrees
                # gaze_pitch_yaw = np.degrees(gaze_pitch_yaw)
                # predictions[i] = np.degrees(predictions[i])

                # convert pitch, yaw to gaze vector
                # gt_gaze_vector = pitch_yaw_to_gaze_vector(*gaze_pitch_yaw[i])
                # pred_gaze_vector = pitch_yaw_to_gaze_vector(*predictions[i])

                vis_image = self.draw_gaze_vector(uint8_image, gt_gaze_vector, color=(0,255,0))  # Ground truth
                vis_image = self.draw_gaze_vector(uint8_image, pred_gaze_vector, color=(255,0,0), thickness=1)  # Prediction
                
                # Write the gt pitch and yaw values on the image
                # gt_pitch, gt_yaw = gaze_pitch_yaw[i]
                # cv2.putText(vis_image, f"GT: {gt_pitch:.2f}, {gt_yaw:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                visualizations.append(vis_image)

            # Log images to TensorBoard
            with self.file_writer.as_default():
                for i, vis in enumerate(visualizations):
                    # tf.summary.image(f"Epoch {epoch} - Sample {i}", vis[np.newaxis], step=epoch)
                    tf.summary.image(f"{self.name} - S{i}", vis[np.newaxis], step=epoch)
            break

class ImageVisCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir, img_size, name='Gaze'):
        super().__init__()
        self.dataset = dataset  # Dataset to visualize predictions
        self.log_dir = log_dir  # Directory to store TensorBoard logs
        self.img_size = img_size
        self.name = name
        self.file_writer = tf.summary.create_file_writer(str(log_dir))

    def on_epoch_end(self, epoch, logs=None):
        # Select a batch from the dataset
        # for images, (gaze_vector, embedding) in self.dataset.take(1):
        for images, _ in self.dataset.take(1):
            output_images = self.model.predict(images)  # Get model predictions
            images = images.numpy()  # Convert images to NumPy array
            # Draw gaze vectors on the images
            visualizations = []
            for i in range(len(images)):

                # Denormalize image
                uint8_image = (images[i] * 255).astype(np.uint8)
                uint8_output_image = (output_images[i] * 255).astype(np.uint8)

                # Concatenate the input and output images
                vis_image = np.concatenate([uint8_image, uint8_output_image], axis=1)
                visualizations.append(vis_image)

            # Log images to TensorBoard
            with self.file_writer.as_default():
                for i, vis in enumerate(visualizations):
                    tf.summary.image(f"{self.name} - S{i}", vis[np.newaxis], step=epoch)
            break

# def angular_loss(y_true, y_pred):
#     """
#     Computes the angular loss between the predicted and true gaze directions.

#     Args:
#         y_true: Ground truth relative gaze vectors, shape (batch_size, 3).
#         y_pred: Predicted gaze directions, shape (batch_size, 3).

#     Returns:
#         A scalar tensor representing the mean angular loss.
#     """
#     # Normalize both vectors to unit length
#     y_true = tf.math.l2_normalize(y_true, axis=1)
#     y_pred = tf.math.l2_normalize(y_pred, axis=1)

#     # Compute cosine similarity
#     cosine_similarity = tf.reduce_sum(y_true * y_pred, axis=1)

#     # Clamp cosine similarity to avoid NaN values from acos
#     cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0 + 1e-8, 1.0 - 1e-8)

#     # Compute angular distance (acos of cosine similarity)
#     angular_distance = tf.acos(cosine_similarity)

#     # Return the mean angular distance as loss
#     return tf.reduce_mean(angular_distance)

# def tf_pitch_yaw_to_gaze_vector(pitch, yaw):
#     """
#     Convert pitch and yaw angles to a gaze vector in 3D space.

#     Args:
#         pitch: Tensor of shape (batch_size,) - Pitch angles in radians.
#         yaw: Tensor of shape (batch_size,) - Yaw angles in radians.

#     Returns:
#         A tensor of shape (batch_size, 3) representing the gaze vectors.
#     """
#     x = tf.cos(pitch) * tf.cos(yaw)
#     y = tf.sin(pitch)
#     z = tf.cos(pitch) * tf.sin(yaw)
#     vector = tf.stack([x, y, z], axis=1)
#     norm_vector = tf.math.l2_normalize(vector, axis=1)
#     return norm_vector

def angular_loss(y_true, y_pred):
    """
    Computes angular error between two 3D unit gaze vectors.
    """
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)

    dot_product = tf.reduce_sum(y_true * y_pred, axis=1)
    dot_product = tf.clip_by_value(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    angle_rad = tf.acos(dot_product)
    angle_deg = angle_rad * (180.0 / tf.constant(np.pi, dtype=tf.float32))

    return tf.reduce_mean(angle_deg)


# def angular_loss(y_true, y_pred):
#     """
#     Computes the angular loss between predicted and true gaze directions using pitch and yaw.

#     Args:
#         y_true: Tensor of shape (batch_size, 2) - True gaze directions (pitch, yaw).
#         y_pred: Tensor of shape (batch_size, 2) - Predicted gaze directions (pitch, yaw).

#     Returns:
#         A scalar tensor representing the mean angular error in degrees.
#     """
#     # Convert pitch and yaw to gaze vectors
#     true_vector = tf_pitch_yaw_to_gaze_vector(y_true[:, 0], y_true[:, 1])
#     pred_vector = tf_pitch_yaw_to_gaze_vector(y_pred[:, 0], y_pred[:, 1])

#     # Compute cosine similarity
#     cosine_similarity = tf.reduce_sum(true_vector * pred_vector, axis=1)

#     # Clip cosine similarity to avoid NaN values in acos
#     cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0 + 1e-8, 1.0 - 1e-8)

#     # Compute angular error in radians and convert to degrees
#     angular_distance = tf.acos(cosine_similarity) * (180.0 / tf.constant(3.141592653589793, dtype=tf.float32))

#     # Return mean angular distance
#     return tf.reduce_mean(angular_distance)

def angular_distance(y_true, y_pred):
    """
    Computes the angular distance between the predicted and true gaze vectors in degrees.

    Args:
        y_true: Ground truth gaze vectors, shape (batch_size, 3).
        y_pred: Predicted gaze vectors, shape (batch_size, 3).

    Returns:
        A scalar tensor representing the mean angular distance in degrees.
    """
    # Normalize both vectors to unit length
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)

    # Compute dot product (cosine similarity)
    dot_product = tf.reduce_sum(y_true * y_pred, axis=1)

    # Clamp values to avoid numerical issues with acos
    cos_theta = tf.clip_by_value(dot_product, -1.0 + 1e-8, 1.0 - 1e-8)

    # Compute angular distance in radians
    angular_distance_rad = tf.acos(cos_theta)

    # Convert radians to degrees
    angular_distance_deg = angular_distance_rad * (180.0 / tf.constant(3.141592653589793, dtype=tf.float32))

    # Return the mean angular distance in degrees
    return tf.reduce_mean(angular_distance_deg)

# def angular_distance(y_true, y_pred):
#     """
#     Computes the angular distance between the predicted and true gaze vectors in degrees.

#     Args:
#         y_true: Ground truth gaze pitch and yaw, shape (batch_size, 2).
#         y_pred: Predicted gaze pitch and yaw, shape (batch_size, 2).

#     Returns:
#         A scalar tensor representing the mean angular distance in degrees.
#     """
#     ## Convert pitch and yaw to gaze vectors
#     y_true = tf_pitch_yaw_to_gaze_vector(y_true[:, 0], y_true[:, 1])
#     y_pred = tf_pitch_yaw_to_gaze_vector(y_pred[:, 0], y_pred[:, 1])

#     # Compute dot product (cosine similarity)
#     dot_product = tf.reduce_sum(y_true * y_pred, axis=1)

#     # Clamp values to avoid numerical issues with acos
#     cos_theta = tf.clip_by_value(dot_product, -1.0 + 1e-8, 1.0 - 1e-8)

#     # Compute angular distance in radians
#     angular_distance_rad = tf.acos(cos_theta)

#     # Convert radians to degrees
#     angular_distance_deg = angular_distance_rad * (180.0 / tf.constant(3.141592653589793, dtype=tf.float32))

#     # Return the mean angular distance in degrees
#     return tf.reduce_mean(angular_distance_deg)

# Parsing function for TFRecord
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'gaze_vector': tf.io.FixedLenFeature([2], tf.float32),
        'head_rotation': tf.io.FixedLenFeature([3], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode and preprocess image
    image = tf.image.decode_image(parsed_example['image'], channels=3)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32) / 255.0

    # Extract gaze vector
    gaze_pitch_yaw = parsed_example['gaze_vector']

    # Extract head rotation
    head_pitch_yaw_roll = parsed_example['head_rotation']

    # Perform normalization
    # image = (image - norm_data['image_mean']) / norm_data['image_std']

    return (apply(image), head_pitch_yaw_roll), gaze_pitch_yaw