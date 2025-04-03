import pathlib
import json

from tqdm import tqdm
import h5py
import tensorflow as tf
import numpy as np
import yaml

from webeyetrack.constants import GIT_ROOT

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'

CWD = pathlib.Path(__file__).parent

def get_dataset_metadata(config):

    if config['dataset']['name'] == 'MPIIFaceGaze':
        h5_file = GENERATED_DATASET_DIR / 'MPIIFaceGaze_entire.h5'
        train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        val_ids = [13, 14]
        test_ids = []

    elif config['dataset']['name'] == 'GazeCapture':
        h5_file = GENERATED_DATASET_DIR / 'GazeCapture_entire.h5'
        
        # Load the GazeCapture participant IDs
        with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
            GAZE_CAPTURE_IDS = json.load(f)

        # Get the ids
        if config['dataset']['gazecapture']['num_of_ids'] > 0:
            GAZE_CAPTURE_IDS = GAZE_CAPTURE_IDS[:config['dataset']['gazecapture']['num_of_ids']]
        else:
            GAZE_CAPTURE_IDS = GAZE_CAPTURE_IDS
        
        # Split the GAZE_CAPTURE_IDS into train, validation, and test sets (80-10-10)
        np.random.seed(config['dataset']['seed'])
        np.random.shuffle(GAZE_CAPTURE_IDS)
        num_participants = len(GAZE_CAPTURE_IDS)
        x, y = config['dataset']['train_val_test_split']
        train_size = int(num_participants * x)
        val_size = int(num_participants * y)
        train_ids = GAZE_CAPTURE_IDS[:train_size]
        val_ids = GAZE_CAPTURE_IDS[train_size:train_size+val_size]
        test_ids = GAZE_CAPTURE_IDS[train_size+val_size:]

    return h5_file, train_ids, val_ids, test_ids

# ------------------------------------------------------------------------------------------------
# Train/Validation/Test dataset loading
# ------------------------------------------------------------------------------------------------

def load_datasets(h5_file, train_ids, val_ids, test_ids, config):
    
    # Prepare datasets
    print(f"Loading datasets from {h5_file}")
    train_dataset, train_dataset_size = load_total_dataset(h5_file, participants=train_ids, config=config)
    val_dataset, val_dataset_size = load_total_dataset(h5_file, participants=val_ids, config=config)
    test_dataset, test_dataset_size = load_total_dataset(h5_file, participants=test_ids, config=config)
    print(f"Train dataset size: {train_dataset_size}, Validation dataset size: {val_dataset_size}, Test dataset size: {test_dataset_size}")

    # Sanity check
    for img, label in train_dataset.take(1):
        print("Image batch shape:", img.shape)
        print("Label batch shape:", label.shape)
        print("Image min/max:", tf.reduce_min(img).numpy(), tf.reduce_max(img).numpy())
        # print("Label values:", label.numpy())

    return train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size

def participant_generator(file, pid, config):
    def _gen():
        with h5py.File(file, 'r') as hf:
            group = hf[str(pid)]
            total = group["pixels"].shape[0]
            max_samples = config['dataset']['gazecapture']['max_per_participant']
            limit = min(max_samples, total) if max_samples else total
            for i in range(limit):
                image = group["pixels"][i].astype(np.float32) / 255.0
                label = group["pog_norm"][i][:2].astype(np.float32)
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
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
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

# ------------------------------------------------------------------------------------------------
# MAML dataset loading
# ------------------------------------------------------------------------------------------------

def participant_task_generator(h5_file, pid, config):
    support_size = config['dataset']['support_size']
    query_size = config['dataset']['query_size']
    
    def _gen():
        with h5py.File(h5_file, 'r') as hf:
            group = hf[str(pid)]
            total = group["pixels"].shape[0]
            indices = np.arange(total)
            np.random.shuffle(indices)

            # Clip support and query sizes to the total number of samples
            safe_support_size = min(support_size, total)
            safe_query_size = min(query_size, total - support_size)

            support_indices = indices[:safe_support_size]
            query_indices = indices[safe_support_size:safe_support_size + safe_query_size]

            def get_samples(idxs):
                for i in idxs:
                    image = group["pixels"][i].astype(np.float32) / 255.0
                    label = group["pog_norm"][i][:2].astype(np.float32)
                    yield image, label

            support = list(get_samples(support_indices))
            query = list(get_samples(query_indices))

        return support, query
    return _gen

def prepare_task_generators(h5_file, participants, config):
    task_generators = {}
    support_size = config['dataset']['support_size']
    query_size = config['dataset']['query_size']
    total_required = support_size + query_size

    # Open file once and keep reference (still lazy loading)
    h5_ref = h5py.File(h5_file, 'r')

    for pid in participants:
        group = h5_ref[str(pid)]
        total = group["pixels"].shape[0]

        if total < total_required:
            print(f"Skipping participant {pid} with only {total} samples (requires {total_required})")
            continue

        def _make_task_fn(pid=pid):
            def sample_task():
                group = h5_ref[str(pid)]
                total = group["pixels"].shape[0]
                indices = np.arange(total)
                np.random.shuffle(indices)

                support_indices = indices[:support_size]
                query_indices = indices[support_size:support_size + query_size]

                def get_samples(idxs):
                    for i in idxs:
                        image = group["pixels"][i].astype(np.float32) / 255.0
                        label = group["pog_norm"][i][:2].astype(np.float32)
                        yield image, label

                support = list(get_samples(support_indices))
                query = list(get_samples(query_indices))
                support_x, support_y = zip(*support)
                query_x, query_y = zip(*query)
                return (
                    tf.stack(support_x), tf.stack(support_y),
                    tf.stack(query_x), tf.stack(query_y)
                )
            return sample_task

        task_generators[pid] = _make_task_fn()

    if len(task_generators) == 0:
        raise ValueError("No valid participants found with enough samples for support + query sets.")

    return task_generators


def task_sampler(h5_file, participants, config):
    while True:
        pid = np.random.choice(participants)
        task_gen = participant_task_generator(h5_file, pid, config)
        support, query = task_gen()
        support_x, support_y = zip(*support)
        query_x, query_y = zip(*query)

        yield (
            tf.stack(support_x), tf.stack(support_y),
            tf.stack(query_x), tf.stack(query_y)
        )

# def task_sampler_sequential(h5_file, participants, config):
#     while True:  # Infinite loop for dataset
#         for pid in participants:
#             task_gen = participant_task_generator(h5_file, pid, config)
#             support, query = task_gen()
#             support_x, support_y = zip(*support)
#             query_x, query_y = zip(*query)

#             yield (
#                 tf.stack(support_x), tf.stack(support_y),
#                 tf.stack(query_x), tf.stack(query_y)
#             )

def fast_task_sampler(task_generators, participants):
    available_pids = list(task_generators.keys())
    if len(available_pids) == 0:
        raise ValueError("No valid participants found with enough samples for support + query sets.")
    while True:
        pid = np.random.choice(available_pids)
        yield task_generators[pid]()

def get_maml_task_dataset(h5_file, participants, config):
    support_size = config['dataset']['support_size']
    query_size = config['dataset']['query_size']
    
    output_signature = (
        tf.TensorSpec(shape=(support_size, *config['model']['input_shape']), dtype=tf.float32),
        tf.TensorSpec(shape=(support_size, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *config['model']['input_shape']), dtype=tf.float32), # None to support variable batch size
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        # lambda: task_sampler(h5_file, participants, config),
        lambda: fast_task_sampler(prepare_task_generators(h5_file, participants, config), participants),
        output_signature=output_signature
    )

if __name__ == '__main__':

    # Load the config
    with open(CWD / 'configs' / 'gaze_gazecapture_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test loading GazeCapture dataset
    h5_file = GENERATED_DATASET_DIR / 'GazeCapture_entire.h5'
    with open(CWD.parent / 'GazeCapture_participant_ids.json', 'r') as f:
        GAZE_CAPTURE_IDS = json.load(f)

    print("Loading dataset...")
    
    dataset, _, size, _, _, _ = load_datasets(
        h5_file, 
        train_ids=GAZE_CAPTURE_IDS[:100],
        val_ids=[GAZE_CAPTURE_IDS[-1]],
        test_ids=[GAZE_CAPTURE_IDS[-1]],
        config=config
    )

    print("Dataset loaded.")

    # Check that all entries in the dataset are valid
    # Essentially, no NaN values

    for img, label in tqdm(dataset, total=size):
        # assert not np.isnan(img.numpy()).any(), "Image contains NaN values"
        # print(label)
        assert not np.isnan(label.numpy()).any(), "Label contains NaN values"
        # print("No NaN values found in the dataset")