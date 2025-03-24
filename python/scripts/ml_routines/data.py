import h5py
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from functools import partial

def participant_generator(file, pid, max_samples=None):
    def _gen():
        with h5py.File(file, 'r') as hf:
            group = hf[str(pid)]
            total = group["pixels"].shape[0]
            limit = min(max_samples, total) if max_samples else total
            for i in range(limit):
                image = group["pixels"][i].astype(np.float32) / 255.0
                label = group["labels"][i][:3].astype(np.float32)
                yield image, label
    return _gen

def load_total_dataset(hdf5_path, participants, batch_size, max_samples_per_participant=None):
    datasets = [
        tf.data.Dataset.from_generator(
            participant_generator(hdf5_path, pid, max_samples_per_participant),
            output_signature=(
                tf.TensorSpec(shape=(128, 512, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
            )
        )
        for pid in participants
    ]

    ds = tf.data.Dataset.sample_from_datasets(datasets, weights=None, seed=42)  # Uniform sampling

    # ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE).repeat()
    # ds.batch(batch_size).cache() \
    #     .shuffle(5000, reshuffle_each_iteration=True) \
    #     .prefetch(tf.data.AUTOTUNE)

    # with h5py.File(hdf5_path, 'r') as hf:
    #     total = sum(hf[str(p)]["pixels"].shape[0] for p in participants)
    with h5py.File(hdf5_path, 'r') as hf:
        total = sum(min(max_samples_per_participant, hf[str(p)]["pixels"].shape[0])
                    if max_samples_per_participant else hf[str(p)]["pixels"].shape[0]
                    for p in participants)

    return ds, total