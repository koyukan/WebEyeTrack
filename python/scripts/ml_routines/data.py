import pathlib
import json
import pickle

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

def tensor_spec(shape): return tf.TensorSpec(shape=shape, dtype=tf.float32)

def get_dataset_metadata(config):

    if config['dataset']['name'] == 'MPIIFaceGaze':
        h5_file = GENERATED_DATASET_DIR / 'MPIIFaceGaze_entire.h5'
        train_ids = config['dataset']['train_ids']
        val_ids = config['dataset']['val_ids']
        test_ids = config['dataset']['test_ids']

    elif config['dataset']['name'] == 'EyeDiap':
        h5_file = GENERATED_DATASET_DIR / 'EyeDiap_entire.h5'
        train_ids = config['dataset']['train_ids']
        val_ids = config['dataset']['val_ids']
        test_ids = config['dataset']['test_ids']

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

        # Add the ids to the config to record them later
        config['dataset']['gazecapture']['train_ids'] = train_ids
        config['dataset']['gazecapture']['val_ids'] = val_ids
        config['dataset']['gazecapture']['test_ids'] = test_ids

    return h5_file, train_ids, val_ids, test_ids

def augment_sample(sample):
    image = sample["image"]

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    sample["image"] = image
    return sample

# ------------------------------------------------------------------------------------------------
# Train/Validation/Test dataset loading
# ------------------------------------------------------------------------------------------------

def load_datasets(h5_file, train_ids, val_ids, test_ids, config):
    
    # Prepare datasets
    print(f"Loading datasets from {h5_file}")
    train_dataset, train_dataset_size = load_total_dataset(h5_file, participants=train_ids, config=config, augment=True)
    val_dataset, val_dataset_size = load_total_dataset(h5_file, participants=val_ids, config=config)
    test_dataset, test_dataset_size = load_total_dataset(h5_file, participants=test_ids, config=config)
    print(f"Train dataset size: {train_dataset_size}, Validation dataset size: {val_dataset_size}, Test dataset size: {test_dataset_size}")

    # Sanity check
    # import pdb; pdb.set_trace()
    # for img, label in train_dataset.take(1):
    #     print("Image batch shape:", img.shape)
    #     print("Label batch shape:", label.shape)
    #     print("Image min/max:", tf.reduce_min(img).numpy(), tf.reduce_max(img).numpy())
    #     # print("Label values:", label.numpy())

    return train_dataset, val_dataset, train_dataset_size, val_dataset_size, test_dataset, test_dataset_size

def participant_generator(file, pid, config, weights_data):
    def _gen():
        with h5py.File(file, 'r') as hf:
            group = hf[str(pid)]
            total = group["pixels"].shape[0]

            # Unpack bin data
            if weights_data is not None:
                bin_weights = weights_data["weights"]
                xedges = weights_data["xedges"]
                yedges = weights_data["yedges"]

            try:
                max_samples = config['dataset']['gazecapture']['max_per_participant']
            except KeyError:
                max_samples = None
            
            limit = min(max_samples, total) if max_samples else total
            for i in range(limit):

                pog_norm = group["pog_norm"][i][:2].astype(np.float32) - np.array([0.5, 0.5])

                if weights_data is not None:
                    # Compute bin indices
                    x_idx = np.digitize([pog_norm[0]], xedges)[0] - 1
                    y_idx = np.digitize([pog_norm[1]], yedges)[0] - 1

                    # Clip to bounds
                    x_idx = np.clip(x_idx, 0, bin_weights.shape[0] - 1)
                    y_idx = np.clip(y_idx, 0, bin_weights.shape[1] - 1)

                    # Fetch weight
                    sample_weight = float(bin_weights[x_idx, y_idx])
                else:
                    sample_weight = 1.0

                yield {
                    "image": group["pixels"][i].astype(np.float32) / 255.0,
                    "head_vector": group["head_vector"][i][:3].astype(np.float32),
                    "face_origin_3d": group["face_origin_3d"][i][:3].astype(np.float32),
                    'screen_info': np.stack([group['screen_height_cm'][i],
                                                group['screen_width_cm'][i]]).astype(np.float32),
                    "pog_norm": pog_norm,
                    "sample_weight": sample_weight
                }
    return _gen

def load_total_dataset(
        hdf5_path, 
        participants, 
        config,
        augment=False,
    ):

    assert config['config']['type'] == 'autoencoder', "Only 'autoencoder' type configuration is allowed."

    # Load weights for sampling if available
    weights_data = None
    try:
        if config['dataset']['weighted']:
            if config['dataset']['name'] == 'GazeCapture':
                bin_pickle = GENERATED_DATASET_DIR / 'GazeCapture_bin_weights.pkl'
            elif config['dataset']['name'] == 'EyeDiap':
                bin_pickle = GENERATED_DATASET_DIR / 'EyeDiap_bin_weights.pkl'
            else:
                raise ValueError("Weighted sampling is only supported for GazeCapture dataset.")
            
            with open(bin_pickle, 'rb') as f:
                weights_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading weights data: {e}")        

    output_signature = {
        "image": tensor_spec((config['model']['encoder']['input_shape'])),  # Image
        "head_vector": tensor_spec((3,)),  # head_vector
        "face_origin_3d": tensor_spec((3,)),  # face_origin_3d
        "screen_info": tensor_spec((2,)),  # screen_info
        "pog_norm": tensor_spec((2,)),  # Normalized point of gaze
        "sample_weight": tensor_spec(())  # Sample weight for weighted sampling
    }
    
    datasets = [
        tf.data.Dataset.from_generator(
            participant_generator(hdf5_path, pid, config, weights_data),
            output_signature=output_signature
        )
        for pid in participants
    ]

    ds = tf.data.Dataset.sample_from_datasets(datasets, weights=None, seed=config['dataset']['seed'])  # Uniform sampling

    if augment:
        ds = ds.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds = ds.shuffle(config['dataset']['shuffle_buffer']) \
        .batch(config['training']['batch_size']) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE).repeat()
    
    # ds.batch(batch_size).cache() \
    #     .shuffle(5000, reshuffle_each_iteration=True) \
    #     .prefetch(tf.data.AUTOTUNE)

    # with h5py.File(hdf5_path, 'r') as hf:
    #     total = sum(hf[str(p)]["pixels"].shape[0] for p in participants)
    try:
        mpp = config['dataset']['gazecapture']['max_per_participant']
    except KeyError:
        mpp = None

    with h5py.File(hdf5_path, 'r') as hf:
        total = sum(min(mpp, hf[str(p)]["pixels"].shape[0])
                    if mpp else hf[str(p)]["pixels"].shape[0]
                    for p in participants)

    return ds, total

# ------------------------------------------------------------------------------------------------
# MAML dataset loading
# ------------------------------------------------------------------------------------------------

def prepare_task_generators(h5_file, participants, config):
    task_generators = {}
    support_size = config['dataset']['support_size']
    query_size = config['dataset']['query_size']
    total_required = support_size + query_size
    h5_ref = h5py.File(h5_file, 'r')
    # pixels_label = config['model']['encoder']['input_name']
    pixels_label = 'pixels'

    for pid in participants:
        group = h5_ref[str(pid)]
        total = group[pixels_label].shape[0]

        if total < total_required:
            print(f"Skipping participant {pid} with only {total} samples (requires {total_required})")
            continue

        def _make_task_fn(pid=pid):
            def sample_task():
                group = h5_ref[str(pid)]
                total = group[pixels_label].shape[0]
                indices = np.random.permutation(total)

                support_indices = indices[:support_size]
                query_indices = indices[support_size:support_size + query_size]

                def get_samples(idxs):
                    for i in idxs:
                        yield {
                            "image": group[pixels_label][i].astype(np.float32) / 255.0,
                            "head_vector": group["head_vector"][i][:3].astype(np.float32),
                            "face_origin_3d": group["face_origin_3d"][i][:3].astype(np.float32),
                            'screen_info': np.stack([group['screen_height_cm'][i],
                                                     group['screen_width_cm'][i]]).astype(np.float32),
                        }, group["pog_norm"][i][:2].astype(np.float32) - np.array([0.5, 0.5])

                support = list(get_samples(support_indices))
                query = list(get_samples(query_indices))

                support_x, support_y = zip(*support)
                query_x, query_y = zip(*query)

                # Stack dict of lists into dict of tensors
                def stack_dict(samples):
                    return {
                        key: tf.stack([d[key] for d in samples])
                        for key in samples[0]
                    }

                return (
                    stack_dict(support_x), tf.stack(support_y),
                    stack_dict(query_x), tf.stack(query_y)
                )
            return sample_task

        task_generators[pid] = _make_task_fn()

    if not task_generators:
        raise ValueError("No valid participants found with enough samples.")

    return task_generators


def fast_task_sampler(task_generators, participants):
    available_pids = list(task_generators.keys())
    if len(available_pids) == 0:
        raise ValueError("No valid participants found with enough samples for support + query sets.")
    while True:
        pid = np.random.choice(available_pids)
        yield task_generators[pid]()

def get_maml_task_dataset(h5_file, participants, config):
    input_shape = config['model']['encoder']['input_shape']
    support_size = config['dataset']['support_size']
    query_size = config['dataset']['query_size']

    input_signature = {
        "image": tensor_spec((None, *input_shape)),
        "head_vector": tensor_spec((None, 3)),
        "face_origin_3d": tensor_spec((None, 3)),
        'screen_info': tensor_spec((None, 2)),
    }

    return tf.data.Dataset.from_generator(
        lambda: fast_task_sampler(
            prepare_task_generators(h5_file, participants, config), participants
        ),
        output_signature=(
            input_signature,                   # support_x
            tensor_spec((support_size, 2)),    # support_y
            input_signature,                   # query_x
            tensor_spec((query_size, 2))       # query_y
        )
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