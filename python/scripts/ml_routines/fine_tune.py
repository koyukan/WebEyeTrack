import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from webeyetrack.constants import GIT_ROOT
from webeyetrack.models.blazegaze import get_gaze_model, init_model

from blazegaze_utils import angular_loss, angular_distance, parse_tfrecord_fn
from aug import apply

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
BATCH_SIZE = 1
IMG_SIZE = 128
EPOCHS = 10
N_SWEEP = [1, 2, 5, 10, 20, 50]  # Number of samples to fine-tune with

VAL_TFRECORD_PATH = GENERATED_DATASET_DIR / 'val_MPIIFaceGaze_blazegaze_p13.tfrecord'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = MODELS_DIR / 'blazegaze_model' / '10-0.17.h5'


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
    return apply(image, gaze_vector)


def load_dataset(tfrecord_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def fine_tune_model(model, train_samples, val_dataset, epochs=EPOCHS):
    """Fine-tunes BlazeGaze on a given number of samples."""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss=angular_loss, 
                  metrics=[angular_distance])
    # model.fit(train_samples, validation_data=val_dataset, epochs=epochs, verbose=1)
    model.fit(train_samples, epochs=epochs, verbose=1)
    return model


# Load validation dataset
val_dataset = load_dataset(VAL_TFRECORD_PATH, BATCH_SIZE)

# Sweep over different numbers of fine-tuning samples
train_angular_errors = []
val_angular_errors = []
for n in N_SWEEP:

    # Load pre-trained model
    model = get_gaze_model()
    init_model(model)
    model.load_weights(MODEL_PATH)

    print(f"Fine-tuning with {n} samples...")
    train_samples = val_dataset.take(n)  # Use first n samples from validation set
    fine_tuned_model = fine_tune_model(model, train_samples, val_dataset)

    # Get the training angular error
    results = fine_tuned_model.evaluate(train_samples, verbose=0)
    train_angular_errors.append(results[1])
    
    # Evaluate performance
    results = fine_tuned_model.evaluate(val_dataset, verbose=0)
    val_angular_errors.append(results[1])

# Plot performance curve
plt.figure()
plt.plot(N_SWEEP, val_angular_errors, marker='o', linestyle='-', label='Validation', color='red')
plt.plot(N_SWEEP, train_angular_errors, marker='o', linestyle='-', label='Training', color='blue')
plt.xlabel('Number of Training Samples')
plt.ylabel('Angular Error (Degrees)')
plt.title('Fine-tuning BlazeGaze with Increasing Samples')
plt.grid()
plt.show()
