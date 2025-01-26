import os
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from webeyetrack.constants import GIT_ROOT
from webeyetrack.models.blazegaze import get_gaze_model, init_model

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
BATCH_SIZE = 1
IMG_SIZE = 128
EPOCHS = 50

TFRECORD_PATH = GENERATED_DATASET_DIR / 'MPIIFaceGaze_blazegaze.tfrecord'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = FILE_DIR / 'models' / 'blazegaze_model.h5'
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

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
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

# Callbacks
checkpoint_callback = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
learning_rate_callback = LearningRateScheduler(lambda epoch: 1e-3 * (0.1 ** (epoch // 10)))

# Train model
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback]
)

# Evaluate model
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test MAE: {results[1]}")
