import h5py
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from functools import partial

def participant_generator(file, pid, config):
    def _gen():
        with h5py.File(file, 'r') as hf:
            group = hf[str(pid)]
            total = group["pixels"].shape[0]
            max_samples = config['dataset']['gazecapture']['max_per_participant']
            limit = min(max_samples, total) if max_samples else total
            for i in range(limit):
                image = group["pixels"][i].astype(np.float32) / 255.0
                label = group["labels"][i][:3].astype(np.float32)
                if config['model']['mode'] == 'autoencoder':
                    yield image, image
                elif config['model']['mode'] == 'gaze':
                    yield image, label
                else:
                    raise ValueError("Invalid mode. Must be either 'autoencoder' or 'gaze'")
    return _gen

def load_total_dataset(
        hdf5_path, 
        participants, 
        config
    ):

    if config['model']['mode'] == 'autoencoder':
        output_signature = (
            tf.TensorSpec(shape=config['model']['input_shape'], dtype=tf.float32),
            tf.TensorSpec(shape=config['model']['input_shape'], dtype=tf.float32),
        )
    elif config['model']['mode'] == 'gaze':
        output_signature = (
            tf.TensorSpec(shape=config['model']['input_shape'], dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
        )
    else:
        raise ValueError("Invalid mode. Must be either 'autoencoder' or 'gaze'")
    
    datasets = [
        tf.data.Dataset.from_generator(
            participant_generator(hdf5_path, pid, config),
            output_signature=output_signature
        )
        for pid in participants
    ]

    ds = tf.data.Dataset.sample_from_datasets(datasets, weights=None, seed=config['dataset']['seed'])  # Uniform sampling

    # ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds = ds.shuffle(1000) \
        .batch(config['training']['batch_size']) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE).repeat()
    
    # ds.batch(batch_size).cache() \
    #     .shuffle(5000, reshuffle_each_iteration=True) \
    #     .prefetch(tf.data.AUTOTUNE)

    # with h5py.File(hdf5_path, 'r') as hf:
    #     total = sum(hf[str(p)]["pixels"].shape[0] for p in participants)
    mpp = config['dataset']['gazecapture']['max_per_participant']
    with h5py.File(hdf5_path, 'r') as hf:
        total = sum(min(mpp, hf[str(p)]["pixels"].shape[0])
                    if mpp else hf[str(p)]["pixels"].shape[0]
                    for p in participants)

    return ds, total