import os
import argparse
import json
import pathlib

from webeyetrack import WebEyeTrack, WebEyeTrackConfig
from webeyetrack.blazegaze import BlazeGazeConfig, build_full_inference_model
import yaml
import cattrs
import tensorflowjs as tfjs

CWD = pathlib.Path(__file__).parent
SAVED_MODELS_DIR = CWD / 'saved_models'
CONFIG_PATH = SAVED_MODELS_DIR / '2025-06-02-16-34-03_mpiifacegaze_maml_full_run' / 'config.yaml'

OUTPUT_DIR = SAVED_MODELS_DIR / 'tfjs'

if __name__ == "__main__":

  # Load the configuration file (YAML)
  with open(CONFIG_PATH, 'r') as f:
      config = yaml.safe_load(f)

  # Create the WebEyeTrack object
  wet = WebEyeTrack(
      WebEyeTrackConfig(
          screen_px_dimensions=(-1, -1),
          screen_cm_dimensions=(-1, -1)
      )
  )

  # Extract the pieces
  encoder_model = wet.blazegaze.encoder
  gaze_mlp = wet.blazegaze.gaze_mlp

  # Load model
  model_config = cattrs.structure(config['model'], BlazeGazeConfig)

  # Create a new model with the weights and biases
  full_model = build_full_inference_model(encoder_model, gaze_mlp, model_config)

  # Save the model as a .h5 file
  output_dir = OUTPUT_DIR / 'full_model'
  output_dir.mkdir(parents=True, exist_ok=True)
  full_model.export(output_dir)

  # Then run the folliwing command
  """
  tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    /path/to/full_model \
    /path/to/output_tfjs_model
  """

  # Convert the model to TensorFlow.js format
  # tfjs.converters.save_keras_model(full_model, OUTPUT_DIR)
