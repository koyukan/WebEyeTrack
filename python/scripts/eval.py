import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets import MPIIFaceGazeDataset
from webeyetrack.pipelines import FLGE

FILE_DIR = pathlib.Path(__file__).parent

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def distance(y, y_hat):
    return np.abs(y_hat - y)

def eval():

    # Create pipeline
    algo = FLGE()
    
    # Create a dataset object
    dataset = MPIIFaceGazeDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['MPIIFaceGaze']['path']),
        img_size=[244,244],
        face_size=[244,244],
        dataset_size=100
    )

    metric_functions = {'depth': distance}
    metrics = defaultdict(list)
    for sample in tqdm(dataset, total=len(dataset)):

        # Process the sample
        output = algo.process_sample(sample)

        # Compute the error
        actual = {
            'depth': sample['face_origin_3d'][2]
        }

        for name, function in metric_functions.items():
            metrics[name].append(function(actual[name], output[name]))

    # Generate box plots for the metrics
    df = pd.DataFrame(metrics)
    sns.boxplot(df)
    plt.show()

if __name__ == '__main__':
    eval()