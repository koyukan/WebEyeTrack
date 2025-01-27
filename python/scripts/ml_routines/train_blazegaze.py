import os
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from webeyetrack.constants import GIT_ROOT
from webeyetrack.models.blazegaze import get_gaze_model, init_model

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
BATCH_SIZE = 32
IMG_SIZE = 128
EPOCHS = 10

TRAIN_TFRECORD_PATH = GENERATED_DATASET_DIR / 'train_MPIIFaceGaze_blazegaze.tfrecord'
VAL_TFRECORD_PATH = GENERATED_DATASET_DIR / 'val_MPIIFaceGaze_blazegaze.tfrecord'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = FILE_DIR / 'models' / 'blazegaze_model.h5'
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

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
        for images, gaze_vectors in self.dataset.take(1):
            predictions = self.model.predict(images)  # Get model predictions
            images = images.numpy()  # Convert images to NumPy array
            gaze_vectors = gaze_vectors.numpy()  # Ground truth gaze vectors
            predictions = predictions  # Predicted gaze vectors

            # Draw gaze vectors on the images
            visualizations = []
            for i in range(len(images)):
                uint8_image = (images[i] * 255).astype(np.uint8)
                uint8_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
                vis_image = self.draw_gaze_vector(uint8_image, gaze_vectors[i], color=(0,255,0))  # Ground truth
                vis_image = self.draw_gaze_vector(uint8_image, predictions[i], color=(255,0,0), thickness=1)  # Prediction
                visualizations.append(vis_image)

            # Log images to TensorBoard
            with self.file_writer.as_default():
                for i, vis in enumerate(visualizations):
                    # tf.summary.image(f"Epoch {epoch} - Sample {i}", vis[np.newaxis], step=epoch)
                    tf.summary.image(f"{self.name} - S{i}", vis[np.newaxis], step=epoch)
            break

def angular_loss(y_true, y_pred):
    """
    Computes the angular loss between the predicted and true gaze directions.

    Args:
        y_true: Ground truth relative gaze vectors, shape (batch_size, 3).
        y_pred: Predicted gaze directions, shape (batch_size, 3).

    Returns:
        A scalar tensor representing the mean angular loss.
    """
    # Normalize both vectors to unit length
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)

    # Compute cosine similarity
    cosine_similarity = tf.reduce_sum(y_true * y_pred, axis=1)

    # Clamp cosine similarity to avoid NaN values from acos
    cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0 + 1e-8, 1.0 - 1e-8)

    # Compute angular distance (acos of cosine similarity)
    angular_distance = tf.acos(cosine_similarity)

    # Return the mean angular distance as loss
    return tf.reduce_mean(angular_distance)

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


# Parsing function for TFRecord
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'gaze_vector': tf.io.FixedLenFeature([3], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode and preprocess image
    image = tf.image.decode_image(parsed_example['image'], channels=3)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32) / 255.0

    # Extract gaze vector
    gaze_vector = parsed_example['gaze_vector']
    return image, gaze_vector

def load_dataset(tfrecord_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

    # # Calculate sizes for train, validation, and test splits
    # total_size = sum(1 for _ in raw_dataset)
    # # train_size = 1
    # # valid_size = 1
    # # test_size = 1
    # train_size = int(total_size * train_split)
    # valid_size = int(total_size * valid_split)
    # test_size = total_size - train_size - valid_size

    # print(f"Total size: {total_size}, Train size: {train_size}, Valid size: {valid_size}, Test size: {test_size}")

    # # Split datasets
    # train_dataset = dataset.take(train_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    # remaining_dataset = dataset.skip(train_size)
    # valid_dataset = remaining_dataset.take(valid_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    # test_dataset = remaining_dataset.skip(valid_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # # Debug shapes of a single batch
    # for input_features in dataset.take(1):
    #     print("Input features shape:", input_features[0].shape)
    #     print("Gaze vector shape:", input_features[1].shape)

    # return train_dataset, valid_dataset, test_dataset

# Prepare datasets
# train_dataset, valid_dataset, test_dataset = load_and_split_dataset(TFRECORD_PATH, BATCH_SIZE)
train_dataset = load_dataset(TRAIN_TFRECORD_PATH, BATCH_SIZE)
val_dataset = load_dataset(VAL_TFRECORD_PATH, BATCH_SIZE)

train_dataset_size = len(list(train_dataset))

# Make a learning rate schedule based on the epoch instead of steps
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=train_dataset_size,
    decay_rate=0.96
)

# Load model
model = get_gaze_model()
init_model(model)
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=angular_loss, metrics=[angular_distance])

# Callbacks
checkpoint_callback = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
# learning_rate_callback = LearningRateScheduler(lambda epoch: 1e-3 * (0.1 ** (epoch // 10)))

# Subset the dataset for visualization (use a few samples)
train_vis_dataset = train_dataset.take(1)
valid_vis_dataset = val_dataset.take(1)

# Define the callback
train_vis_callback = GazeVisualizationCallback(
    dataset=train_vis_dataset,
    log_dir=LOG_PATH / "visualizations",
    img_size=IMG_SIZE,
    name='Gaze (Training)'
)
valid_vis_callback = GazeVisualizationCallback(
    dataset=valid_vis_dataset,
    log_dir=LOG_PATH / "visualizations",
    img_size=IMG_SIZE,
    name='Gaze (Validation)'
)

log_dir = LOG_PATH / 'images'
os.makedirs(log_dir, exist_ok=True)

# Train model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_callback, 
        tensorboard_callback, 
        train_vis_callback,
        valid_vis_callback
    ]
)

# Evaluate model
# results = model.evaluate(test_dataset)
# print(f"Test Loss: {results[0]}, Test Angular Error (Degrees): {results[1]}")
