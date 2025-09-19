# WebEyeTrack in Python

Created by <a href="https://edavalosanaya.github.io" target="_blank">Eduardo Davalos</a>, <a href="https://scholar.google.com/citations?user=_E0SGAkAAAAJ&hl=en" target="_blank">Yike Zhang</a>, <a href="https://scholar.google.com/citations?user=GWvdYIoAAAAJ&hl=en&oi=ao" target="_blank">Namrata Srivastava</a>, <a href="https://www.linkedin.com/in/yashvitha/" target="_blank">Yashvitha Thatigolta</a>, <a href="" target="_blank">Jorge A. Salas</a>, <a href="https://www.linkedin.com/in/sara-mcfadden-93162a4/" target="_blank">Sara McFadden</a>, <a href="https://scholar.google.com/citations?user=0SHxelgAAAAJ&hl=en" target="_blank">Cho Sun-Joo</a>, <a href="https://scholar.google.com/citations?user=dZ8X7mMAAAAJ&hl=en" target="_blank">Amanda Goodwin</a>, <a href="https://sites.google.com/view/ashwintudur/home" target="_blank">Ashwin TS</a>, and <a href="https://scholar.google.com/citations?user=-m5wrTkAAAAJ&hl=en" target="_blank">Guatam Biswas</a> from <a href="https://wp0.vanderbilt.edu/oele/" target="_blank">Vanderbilt University</a>, <a href="https://redforestai.github.io" target="_blank">Trinity University</a>, and <a href="https://knotlab.github.io/KnotLab/" target="_blank">St. Mary's University</a>

### [Project](https://redforestai.github.io/WebEyeTrack) | [Paper](https://arxiv.org/abs/2508.19544) | [Demo](https://azure-olympie-5.tiiny.site)

<p></p>

[![NPM Version](https://img.shields.io/npm/v/webeyetrack)](https://www.npmjs.com/package/webeyetrack) [![PyPI - Version](https://img.shields.io/pypi/v/webeyetrack)](https://pypi.org/project/webeyetrack/) [![GitHub License](https://img.shields.io/github/license/RedForestAI/webeyetrack)](#license)

As detailed in our paper, the Python version of WebEyeTrack performs the two-step training process: (1) autoencoder and (2) MAML. For first step of training aims to formulate a feature extractor for the gaze embedding that is gaze-aware and effective at translating to PoG estimation. In the second training step performs Model Agnostic Meta-Learning (MAML) to help transform the gaze estimation model into a meta-learner that is more capable of adapting to new unseen persons.

# Installation & Dependencies

For the following repo, make sure to use Tensorflow/Keras 2.x and not the latest Tensorflow/Keras 3.x since the latest version of Keras isn't fully support in TensorflowJS and results in issues when trying to use the required ``LayersModel`` for on-device model adaptation.

Tested setup:
* Ubuntu 24.04
* Python 3.10
* Tensorflow 2.11.1
* TensorflowJS 3.21.0

Aside from this warning, run the following command to install the ``webeyetrack`` package

```bash
git clone https://github.com/redforestai/webeyetrack
cd python
pip install .
```

# Demo

The demo can be located within the ``demo`` directory. This demo uses OpenCV to visualize the webcam and a 1:1 display of the screen location in an image. Make sure to install the ``webeyetrack`` package first. To run the demo, run the following command:

```bash
cd demo
pip install requirements.txt
python main.py
```

# Usage

To use the ``webeyetrack`` package, you need to create a ``WebEyeTrack`` tracker instance.

```python
from webeyetrack import WebEyeTrack, WebEyeTrackConfig
wet = WebEyeTrack(
    WebEyeTrackConfig(
        # Add your screen's meta information here
        # You can use ``screeninfo`` (Windows/Linux) 
        # or ``Quartz`` for MacOS
        screen_px_dimensions=(1920, 1080),
        screen_cm_dimensions=(32, 18),
    )
)
```

Below is a simple example of reading from the webcam via OpenCV and processing the frame.

```python
from webeyetrack import TrackingStatus
cap = cv2.VideoCapture(0)
while True:
    ret, frame = self.cap.read()
    if ret:
        # Process the frame with WebEyeTrack
        status, gaze_result, detection = self.wet.process_frame(frame)

        if status == TrackingStatus.SUCCESS:
            # If successful, perform your logic here
            print(gaze_result.norm_pog)
```

Perform new person adaptation, use the one of the ``adapt`` methods, such as ``adapt_from_gaze_results``. Simply provide a list of prior ``gaze_results`` with their ground truth normalized PoG. 

```python
gaze_results = [...] # list of prior gaze results
norm_pogs = [...] # list of matching groudn truth PoG
wet.adapt_from_gaze_results(gaze_results, norm_pogs)
```

The ``WebEyeTrack`` object returns a ``GazeResults`` object with a status regarding the tracking of the persons' face. Below is a break down of the attributes of ``GazeResults``:

### GazeResult Data Model

| Field Name         | Type                     | Shape / Values        | Description                                         |
|--------------------|--------------------------|------------------------|-----------------------------------------------------|
| facial_landmarks   | `np.ndarray`             | [N, 5]                | Detected facial landmarks                          |
| face_rt            | `np.ndarray`             | [4, 4]                | Face rotation-translation matrix                   |
| face_blendshapes   | `np.ndarray`             | [N, 1]                | Blendshape coefficients for facial expression      |
| eye_patch          | `np.ndarray`             | [H, W, 3]             | RGB image of the eye region                        |
| head_vector        | `np.ndarray`             | [3,]                  | Head direction vector in camera coordinates        |
| face_origin_3d     | `np.ndarray`             | [3,] (X, Y, Z)        | 3D coordinates of the face origin                  |
| metric_face        | `np.ndarray`             | [N, 486]              | Reconstructed 3D face mesh                         |
| metric_transform   | `np.ndarray`             | [4, 4]                | Transformation matrix applied to metric face       |
| gaze_state         | `'open'` or `'closed'`   | Literal               | Blink state of the eye                             |
| norm_pog           | `np.ndarray`             | [2,] (X, Y)           | Normalized point-of-gaze on the screen             |
| durations          | `dict[str, float]`       | —                     | Metadata timing for different processing stages    |

### TrackingStatus Enum

| Value     | Meaning        |
|-----------|----------------|
| 0         | FAILED         |
| 1         | SUCCESS        |

The normalized PoG is from range ``[[-0.5, 0.5], [-0.5, 0.5]]`` where the origin ``(0,0)`` is located at the center of the screen. The positive Y axis is pointing downwards and the positive X axis is pointing toward the right.

# Model Training and Evaluation

For the training, we download the original datasets provided by MPIIFaceGaze, EyeDiap, GazeCapture, and Eye of the Typer (WebGazer) and performed our own preprocessing to ensure that our data normalization was possible to match in the browser. Methods such as ``cv2.solvePnP`` are challenging to translate and therefore should be avoided.

## Datasets

Before starting to download the datasets, please make sure to have more than 1.25 TB available to account for all datasets. Account that you will be downloading a zip and unzipping will increase memory twice over.

Download the following original datasets using the links below:

* [GazeCapture](https://gazecapture.csail.mit.edu/download.php): ~200 GB
* [MPIIFaceGaze](https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/): ~6 GB
* [EyeDiap](https://www.idiap.ch/en/scientific-research/data/eyediap):
* [Eye of the Typer](https://webgazer.cs.brown.edu/data/):

For GazeCapture, MPIIFaceGaze, and EyeDiap, place these datasets within the ``data`` directory at the root of the GitHub repository. 

For Eye of the Typer, this dataset is more challenging to perform the preprocessing and needs to be placed within the WebGazer repository to prevent failing relative imports. Make sure to git clone the [WebGazer repository](https://github.com/brownhci/WebGazer) and place the Eye of the Typer dataset within the ``www/data/src`` directory. 

## Preprocessing

Similar to other gaze estimation methods, we perform a data normalization step to isolate key features (eye patch, head vector, and face position) to make the model training faster and more efficient. However, many of the prior preprocessing steps were either too slow or dependent on functions that aren't easily portable to JavaScript. Therefore, we have modified the data normalization from [Park et al. 2019](https://github.com/NVlabs/few_shot_gaze) to make it faster and real-time.

To perform the preprocessing for GazeCapture, MPIIFaceGaze, and EyeDiap, run the following commands:

```bash
cd python/scripts/preprocessing
python preprocess.py --dataset MPIIFaceGaze
python output_space.py --dataset MPIIFaceGaze
python preprocess.py --dataset GazeCapture
python output_space.py --dataset GazeCapture
python preprocess.py --dataset EyeDiap
python output_space.py --dataset EyeDiap
```

Beware that this preprocessing routines can take up to hours and will consume more memory. The resulting preprocessing datasets will be stored in the ``data/generated`` directory.

For Eye of the Typer dataset preprocessing, following the [instructions provided in the dataset webpage](https://webgazer.cs.brown.edu/data/#:~:text=Using%20the%20Dataset%20Extractor).

## Training Routines

For training, we use configuration files to specify the runs parameters and keep record of the config of prior runs, as this is stored with the logs. The training losses, metrics, and resulting trained models will be stored within the ``logs`` directory inside the ``ml_routines`` via Tensorboard format. Below is how to run the training for all image datasets:

#### autoencoder
```bash
cd python/scripts/ml_routines
python train_autoencoder.py --config configs/train/autoencoder_mpiifacegaze_config.yml
python train_autoencoder.py --config configs/train/autoencoder_gazecapture_config.yml
python train_autoencoder.py --config configs/train/autoencoder_eyediap_config.yml
```
#### maml
For maml training, make sure to change the ``weight_fp`` parameter to point to your autoencoder checkpoint. To make this work, place the generated log directory inside the ``saved_models`` directory inside ``ml_routines`` allow prior checkpoint loading.
```bash
cd python/scripts/ml_routines
python train_maml.py --config configs/train/maml_mpiifacegaze_config.yml
python train_maml.py --config configs/train/maml_gazecapture_config.yml
python train_maml.py --config configs/train/maml_eyediap_config.yml
```

To observe the training logs, run tensorboard with the following command:

```bash
cd python/scripts/ml_routines
python tensorboard --logdir logs
```

## Evaluation Routine

Similar to training, the evaluation uses configuration files to help explore parameters. Make sure to move the trained model directories inside the ``saved_model`` directory and modify the config files ``weight_fp`` to properly load the trained model checkpoint for the purposes of evaluation. To run the evaluations, use the following commands:

```bash
# Image Datasets
cd python/scripts/ml_routines
python eval_maml.py --config configs/eval/maml_mpiifacegaze_config.yml
python eval_maml.py --config configs/eval/maml_gazecapture_config.yml
python eval_maml.py --config configs/eval/maml_eyediap_config.yml
```

# Acknowledgements

The research reported here was supported by the Institute of Education Sciences, U.S. Department of Education, through Grant R305A150199 and R305A210347 to Vanderbilt University. The opinions expressed are those of the authors and do not represent views of the Institute or the U.S. Department of Education.

# Reference

If you use this work in your research, please cite us using the following:

```bibtex
@misc{davalos2025webeyetrack,
	title={WEBEYETRACK: Scalable Eye-Tracking for the Browser via On-Device Few-Shot Personalization},
	author={Eduardo Davalos and Yike Zhang and Namrata Srivastava and Yashvitha Thatigotla and Jorge A. Salas and Sara McFadden and Sun-Joo Cho and Amanda Goodwin and Ashwin TS and Gautam Biswas},
	year={2025},
	eprint={2508.19544},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	url={https://arxiv.org/abs/2508.19544}
}
```

# License

WebEyeTrack is open-sourced under the [MIT License](LICENSE), which permits personal, academic, and commercial use with proper attribution. Feel free to use, modify, and distribute the project.