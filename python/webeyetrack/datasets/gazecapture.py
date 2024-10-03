import pathlib
from dataclasses import asdict
from typing import List, Dict, Union, Tuple, Optional
import shutil
import json

from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2
from PIL import Image
import scipy.io
import yaml
import numpy as np
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..constants import GIT_ROOT
from ..vis import draw_gaze_origin
from .data_protocols import Annotations, CalibrationData, Sample
from .utils import resize_annotations, resize_intrinsics, draw_landmarks_on_image, compute_uv_texture

CWD = pathlib.Path(__file__).parent

class GazeCaptureDataset(Dataset):

    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        dataset_size: Optional[int] = None,
    ):
        
        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        self.dataset_dir = dataset_dir
        assert self.dataset_dir.is_dir(), f"Dataset directory {self.dataset_dir} does not exist."
        self.dataset_size = dataset_size

        # Perform pre-processing
        gz_elements = [x for x in self.dataset_dir.iterdir() if x.suffix == '.gz']
        for gz_element in tqdm(gz_elements, total=len(gz_elements)):
            # If element is a .tar.gz file, extract it
            if gz_element.suffix == '.gz':
                shutil.unpack_archive(gz_element, extract_dir=self.dataset_dir)
                gz_element.unlink()

    def __getitem__(self, idx):
        return {}

    def __len__(self):
        return self.dataset_size
    
if __name__ == "__main__":
    
    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = GazeCaptureDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['GazeCapture']['path']),
        dataset_size=1000,
    )
    print(len(dataset))

    sample = dataset[0]
    print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))