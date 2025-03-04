import os
import pathlib
import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from webeyetrack.constants import GIT_ROOT
from webeyetrack.blazegaze import BlazeGaze
from webeyetrack.tf_utils import angular_loss, angular_distance, parse_tfrecord_fn, GazeVisualizationCallback
from data import load_total_dataset

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

# Constants
BATCH_SIZE = 16
IMG_SIZE = 128
EPOCHS = 20

H5_FILE = GENERATED_DATASET_DIR / 'MPIIFaceGaze.h5'
MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_DIR = MODELS_DIR / TIMESTAMP
os.makedirs(RUN_DIR, exist_ok=True)

# Prepare datasets
full_dataset, dataset_size = load_total_dataset(H5_FILE)
print(f"Dataset size: {dataset_size}")

# Split the dataset
train_size = int(0.8 * dataset_size)
val_size = int(0.2 * dataset_size)

# Shuffle and split
full_dataset = full_dataset.shuffle(dataset_size, seed=42)
train_dataset = full_dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = full_dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

train_dataset_size = len(list(train_dataset))

# Make a learning rate schedule based on the epoch instead of steps
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=train_dataset_size,
    decay_rate=0.96
)

def loss_fn(y_true, y_pred):
    return angular_loss(y_true, y_pred)

# Load model
model = BlazeGaze()
model.tf_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss_fn) #, metrics=[angular_distance])

# Callbacks
# checkpoint_callback = ModelCheckpoint(
#     filepath=RUN_DIR/"blazegaze-{epoch:02d}-{val_loss:.2f}.h5", 
#     monitor="epoch_angular_distance",
#     ave_best_only=True, 
#     save_weights_only=True,
# )
# tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
# learning_rate_callback = LearningRateScheduler(lambda epoch: 1e-3 * (0.1 ** (epoch // 10)))

# # Subset the dataset for visualization (use a few samples)
# train_vis_dataset = train_dataset.take(1)
# valid_vis_dataset = val_dataset.take(1)

# Define the callback
# train_vis_callback = GazeVisualizationCallback(
#     dataset=train_vis_dataset,
#     log_dir=LOG_PATH / "visualizations",
#     img_size=IMG_SIZE,
#     name='Gaze (Training)'
# )
# valid_vis_callback = GazeVisualizationCallback(
#     dataset=valid_vis_dataset,
#     log_dir=LOG_PATH / "visualizations",
#     img_size=IMG_SIZE,
#     name='Gaze (Validation)'
# )

log_dir = LOG_PATH / 'images'
os.makedirs(log_dir, exist_ok=True)

# Train model
model.tf_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        # checkpoint_callback, 
        # tensorboard_callback, 
        # train_vis_callback,
        # valid_vis_callback
    ]
)

# Evaluate model
# results = model.evaluate(test_dataset)
# print(f"Test Loss: {results[0]}, Test Angular Error (Degrees): {results[1]}")
