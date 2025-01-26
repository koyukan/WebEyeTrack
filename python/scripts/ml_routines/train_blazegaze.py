import os
import pathlib
import io

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

TFRECORD_PATH = GENERATED_DATASET_DIR / 'MPIIFaceGaze_blazegaze_full.tfrecord'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = FILE_DIR / 'models' / 'blazegaze_model.h5'
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def log_images(epoch, images, writer):
    figure = plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    with writer.as_default():
        tf.summary.image("Training sample", plot_to_image(figure), step=epoch)

class ImageLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, sample_images):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.sample_images = sample_images

    def on_epoch_end(self, epoch, logs=None):
        log_images(epoch, self.sample_images, self.writer)

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

def load_and_split_dataset(tfrecord_path, batch_size, train_split=0.8, valid_split=0.1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)

    # Calculate sizes for train, validation, and test splits
    total_size = sum(1 for _ in raw_dataset)
    train_size = int(total_size * train_split)
    valid_size = int(total_size * valid_split)
    test_size = total_size - train_size - valid_size

    print(f"Total size: {total_size}, Train size: {train_size}, Valid size: {valid_size}, Test size: {test_size}")

    # Split datasets
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    remaining_dataset = dataset.skip(train_size)
    valid_dataset = remaining_dataset.take(valid_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = remaining_dataset.skip(valid_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Debug shapes of a single batch
    for input_features in dataset.take(1):
        print("Input features shape:", input_features[0].shape)
        print("Gaze vector shape:", input_features[1].shape)

    return train_dataset, valid_dataset, test_dataset

# Prepare datasets
train_dataset, valid_dataset, test_dataset = load_and_split_dataset(TFRECORD_PATH, BATCH_SIZE)

# Load model
model = get_gaze_model()
init_model(model)
model.compile(optimizer=Adam(learning_rate=1e-3), loss=angular_loss, metrics=[angular_distance])

# Callbacks
checkpoint_callback = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
learning_rate_callback = LearningRateScheduler(lambda epoch: 1e-3 * (0.1 ** (epoch // 10)))

log_dir = LOG_PATH / 'images'
os.makedirs(log_dir, exist_ok=True)
# sample_images = next(iter(valid_dataset))[0][:25] # Take a sample of images from the validation set
# image_callback = ImageLoggingCallback(log_dir, sample_images)

# Train model
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback]
)

# Evaluate model
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Angular Error (Degrees): {results[1]}")
