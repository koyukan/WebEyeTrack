import os
import argparse
import json
import pathlib

import tensorflow as tf

# from webeyetrack import WebEyeTrack, WebEyeTrackConfig
# from webeyetrack.blazegaze import BlazeGazeConfig, build_full_inference_model
import yaml
import cattrs
import tensorflowjs as tfjs

"""
# Required fixed for ```ImportError: cannot import name 'shape_poly' from 'jax.experimental.jax2tf'```
https://github.com/jax-ml/jax/issues/18978#issuecomment-2764096155

Fix error in layers
https://github.com/tensorflow/tfjs/issues/2442#issuecomment-563319357
"""

CWD = pathlib.Path(__file__).parent
SAVED_MODELS_DIR = CWD / 'saved_models'
CONFIG_PATH = SAVED_MODELS_DIR / '2025-06-02-16-34-03_mpiifacegaze_maml_full_run' / 'config.yaml'

OUTPUT_DIR = SAVED_MODELS_DIR / 'tfjs'
MODEL_WEIGHTS = pathlib.Path("/media/nicole/T9/GitHub/RedForestAI/WebEyeTrack/python/webeyetrack/model_weights")

from typing import Optional, Literal, List
from dataclasses import dataclass, field
import numpy as np

from webeyetrack.blazegaze import BlazeGaze, BlazeGazeConfig, build_full_inference_model

import tensorflow as tf

if __name__ == "__main__":

    # Load the configuration file (YAML)
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        # config = json.load(f)

    # Modify the weights_fp
    config['model']['weights_fp'] = None

    # Load model
    model_config = cattrs.structure(config['model'], BlazeGazeConfig)
    print(model_config)

    blazegaze = BlazeGaze(model_config)
    encoder_model = blazegaze.encoder
    gaze_mlp = blazegaze.gaze_mlp

    # Create a new model with the weights and biases
    full_model = build_full_inference_model(encoder_model, gaze_mlp, model_config)
    
    # Test passing data through the model
    dummy_inputs = [
        tf.random.uniform((1, *model_config.encoder.input_shape))
    ] + [
        tf.random.uniform((1, *inp.input_shape)) for inp in model_config.gaze.inputs
    ]
    dummy_outputs = full_model(dummy_inputs)
    print("Dummy outputs shape:", [out.shape for out in dummy_outputs])
    
    # Save the model in Keras H5 format
    # full_model.save(OUTPUT_DIR / 'full_model_ATTEMP6.h5')

    # Then run the folliwing command
    """
    tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model full_model_ATTEMP6.h5 full_model/web_layer8
    """

    # Convert the model to TensorFlow.js format
    tfjs.converters.save_keras_model(full_model, OUTPUT_DIR / 'full_model' / 'web')
