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

from blazegaze_utils import angular_loss, angular_distance, parse_tfrecord_fn, GazeVisualizationCallback

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
BATCH_SIZE = 32
IMG_SIZE = 128
EPOCHS = 20

TRAIN_TFRECORD_PATH = GENERATED_DATASET_DIR / 'train_MPIIFaceGaze_blazegaze.tfrecord'
VAL_TFRECORD_PATH = GENERATED_DATASET_DIR / 'val_MPIIFaceGaze_blazegaze.tfrecord'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

def load_dataset(tfrecord_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Prepare datasets
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
checkpoint_callback = ModelCheckpoint(
    filepath=MODELS_DIR/"blazegaze-{epoch:02d}-{val_loss:.2f}.h5", 
    monitor="epoch_angular_distance",
    ave_best_only=True, 
    save_weights_only=True,
)
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
