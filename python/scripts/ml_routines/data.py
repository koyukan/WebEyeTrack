import h5py
import tensorflow as tf

class generator:
    def __init__(self, file, pairs=True):
        """
        :param file: The path to the HDF5 file
        :param pairs: If True, the generator will yield pairs of samples. If False, it will yield single samples
        """
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for group in hf:
                for i in range(hf[group]["pixels"].shape[0]):
                    image = hf[group]["pixels"][i]
                    labels = hf[group]["labels"][i]

                    # Split the labels into gaze and head vectors
                    gaze_labels = labels[:3]
                    head_labels = labels[3:]

                    # Placeholder embedding
                    embedding = tf.zeros(128, dtype=tf.float32)

                    yield (image, (gaze_labels, embedding))

def load_total_dataset(hdf5_path):

    # Determine the size of the hdf5 file
    with h5py.File(hdf5_path, 'r') as hf:
        total = 0
        for group in hf:
            total += hf[group]["pixels"].shape[0]

    ds = tf.data.Dataset.from_generator(
        generator(hdf5_path),
        output_signature=(
            tf.TensorSpec(shape=(128, 512, 3), dtype=tf.uint8),
            (
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
                tf.TensorSpec(shape=(128,), dtype=tf.float32),
            )
        )
    )
    return ds, total