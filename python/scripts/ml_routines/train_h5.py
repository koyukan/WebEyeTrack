import os
import pathlib
import datetime
import json

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

# DATASET = 'MPIIFaceGaze' # 'MPIIFaceGaze' or 'GazeCapture'
DATASET = 'GazeCapture' # 'MPIIFaceGaze' or 'GazeCapture'
if DATASET == 'MPIIFaceGaze':
    H5_FILE = GENERATED_DATASET_DIR / 'MPIIFaceGaze_entire.h5'
    TRAIN_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    VAL_IDS = [13, 14]
    TEST_IDS = []
elif DATASET == 'GazeCapture':
    H5_FILE = GENERATED_DATASET_DIR / 'GazeCapture_entire.h5'
    # Load the GazeCapture participant IDs
    with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
        GAZE_CAPTURE_IDS = json.load(f)

    # For testing, only using 10 participants
    GAZE_CAPTURE_IDS = GAZE_CAPTURE_IDS[:100]
    MAX_PER_PARTICIPANT = BATCH_SIZE * 3
    
    # Split the GAZE_CAPTURE_IDS into train, validation, and test sets (80-10-10)
    np.random.seed(42)
    np.random.shuffle(GAZE_CAPTURE_IDS)
    num_participants = len(GAZE_CAPTURE_IDS)
    train_size = int(num_participants * 0.8)
    val_size = int(num_participants * 0.1)
    TRAIN_IDS = GAZE_CAPTURE_IDS[:train_size]
    VAL_IDS = GAZE_CAPTURE_IDS[train_size:train_size+val_size]
    TEST_IDS = GAZE_CAPTURE_IDS[train_size+val_size:]

MODELS_DIR = FILE_DIR / 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
LOG_PATH = FILE_DIR / 'logs'
os.makedirs(LOG_PATH, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_DIR = MODELS_DIR / TIMESTAMP
os.makedirs(RUN_DIR, exist_ok=True)

def load_datasets():
    
    # Prepare datasets
    print(f"Loading datasets from {H5_FILE}")
    train_dataset, train_dataset_size = load_total_dataset(H5_FILE, participants=TRAIN_IDS, batch_size=BATCH_SIZE, max_samples_per_participant=MAX_PER_PARTICIPANT)
    val_dataset, val_dataset_size = load_total_dataset(H5_FILE, participants=VAL_IDS, batch_size=BATCH_SIZE, max_samples_per_participant=MAX_PER_PARTICIPANT)
    test_dataset, test_dataset_size = load_total_dataset(H5_FILE, participants=TEST_IDS, batch_size=BATCH_SIZE, max_samples_per_participant=MAX_PER_PARTICIPANT)
    print(f"Train dataset size: {train_dataset_size}, Validation dataset size: {val_dataset_size}, Test dataset size: {test_dataset_size}")
    # train_dataset = train_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
    # val_dataset = val_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
    # test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

    # Sanity check
    for img, label in train_dataset.take(1):
        print("Image batch shape:", img.shape)
        print("Label batch shape:", label.shape)
        print("Image min/max:", tf.reduce_min(img).numpy(), tf.reduce_max(img).numpy())
        # print("Label values:", label.numpy())

    return train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size

train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size = load_datasets()

# Make a learning rate schedule based on the epoch instead of steps
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=train_dataset_size,
    decay_rate=0.96
)

def loss_fn(y_true, y_pred):
    import pdb; pdb.set_trace()
    return angular_loss(y_true, y_pred)

# Load model
model = BlazeGaze()
model.tf_model.compile(
    optimizer=Adam(learning_rate=lr_schedule), 
    loss={
        "gaze_output_norm": angular_loss,
        # "embedding_output": "cosine_similarity"
    },
    metrics={
        "gaze_output_norm": angular_distance
    }
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=RUN_DIR/"blazegaze-{epoch:02d}-{val_loss:.2f}.h5", 
    monitor="epoch_angular_distance",
    ave_best_only=True, 
    save_weights_only=True,
)
tensorboard_callback = TensorBoard(
    log_dir=LOG_PATH, 
    # histogram_freq=1, 
    # profile_batch='5,10'
)
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

steps_per_epoch = train_dataset_size // BATCH_SIZE
validation_steps = val_dataset_size // BATCH_SIZE

# Train model
model.tf_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint_callback, 
        tensorboard_callback, 
        train_vis_callback,
        valid_vis_callback
    ]
)

# Evaluate model
# results = model.tf_model.evaluate(test_dataset)
# print(results)
# print(f"Test Loss: {results[0]}, Test Angular Error (Degrees): {results[1]}")
