import h5py
import tensorflow as tf
from tqdm import tqdm

class generator:
    def __init__(self, file, participants=None):
        """
        :param file: The path to the HDF5 file
        :param pairs: If True, the generator will yield pairs of samples. If False, it will yield single samples
        """
        self.file = file
        self.participants = participants

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            # for group in hf:
            for par in self.participants:
                group = hf[str(par)]
                for i in range(group["pixels"].shape[0]):
                    image = group["pixels"][i]
                    labels = group["labels"][i]

                    # Convert uint8 to float32
                    image = tf.cast(image, tf.float32) / 255.0

                    # Split the labels into gaze and head vectors
                    gaze_labels = labels[:3]
                    head_labels = labels[3:]

                    # Placeholder embedding
                    # embedding = tf.zeros(128, dtype=tf.float32)

                    # yield (image, (gaze_labels, embedding))
                    yield (image, gaze_labels)

def load_total_dataset(hdf5_path, participants=None):

    # Determine the size of the hdf5 file
    with h5py.File(hdf5_path, 'r') as hf:
        total = 0
        for group in tqdm(hf, desc="Calculating dataset size"):
            if ((int(group) not in participants) and (group not in participants)):
                continue
            total += hf[group]["pixels"].shape[0]

    ds = tf.data.Dataset.from_generator(
        generator(hdf5_path, participants),
        output_signature=(
            tf.TensorSpec(shape=(128, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
            # (
            #     tf.TensorSpec(shape=(3,), dtype=tf.float32),
            #     tf.TensorSpec(shape=(128,), dtype=tf.float32),
            # )
        )
    )
    return ds, total