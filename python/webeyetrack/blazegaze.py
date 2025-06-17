from typing import Optional, Literal, List
from dataclasses import dataclass, field
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer, 
    Input, 
    DepthwiseConv2D, 
    Conv2D, 
    MaxPool2D, 
    Add, 
    Activation, 
    Flatten,
    Concatenate, 
    Conv2DTranspose,
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Reshape,
    Lambda
)

# Optional: Enable XLA for optimization
tf.config.optimizer.set_jit(True)

from .constants import MODEL_WEIGHTS

@dataclass
class ModalityInput:
    name: str
    input_shape: tuple

@dataclass
class EncoderConfig:
    input_shape: tuple = (128, 512, 3)
    embedding_size: int = 512

@dataclass
class DecoderConfig:
    input_shape: tuple = (8, 32, 96)
    output_shape: tuple = (128, 512, 3)

@dataclass
class GazeConfig:
    inputs: List[ModalityInput] = field(default_factory=lambda: [])

@dataclass
class BlazeGazeConfig:

    # Mode
    mode: Literal['autoencoder', 'maml'] = 'maml'
    weights_fp: Optional[str] = None

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig())
    gaze: GazeConfig = field(default_factory=lambda: GazeConfig())

# ------------------------------------------------------------------------
# Encoder
# ------------------------------------------------------------------------

def blaze_block(y, filters, stride=1):
    x = DepthwiseConv2D((5,5), strides=stride, padding="same")(y)
    x = Conv2D(filters, (1,1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2,2))(y)
        y = Conv2D(filters, (1,1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)

def double_blaze_block(y, filters, stride=1):
    x = DepthwiseConv2D((5,5), strides=stride, padding="same")(y)
    x = Conv2D(filters[0], (1,1), padding="same")(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((5,5), padding="same")(x)
    x = Conv2D(filters[1], (1,1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2,2))(y)
        y = Conv2D(filters[1], (1,1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)

def get_cnn_encoder(config: BlazeGazeConfig):
    x = Input(shape=config.encoder.input_shape)

    # Feature extraction layers
    first_conv = Conv2D(24, (5,5), strides=2, padding="same", activation="relu")(x)
    single_1 = blaze_block(first_conv, 24)
    single_2 = blaze_block(single_1, 24)
    single_3 = blaze_block(single_2, 48, 2)
    single_4 = blaze_block(single_3, 48)
    single_5 = blaze_block(single_4, 48)
    double_1 = double_blaze_block(single_5, [24, 96], 2)
    double_2 = double_blaze_block(double_1, [24, 96])
    double_3 = double_blaze_block(double_2, [24, 96])
    double_4 = double_blaze_block(double_3, [24, 96], 2)
    double_5 = double_blaze_block(double_4, [24, 96])
    double_6 = double_blaze_block(double_5, [24, 96])

    # 🔽 Add gradual squeezing via Conv2D
    z = Conv2D(64, (3, 3), strides=2, padding="same", activation='relu')(double_6)   # → (4, 16, 64)
    z = BatchNormalization()(z)
    z = Conv2D(32, (3, 3), strides=2, padding="same", activation='relu')(z)         # → (2, 8, 32)
    z = BatchNormalization()(z)
    # z = Conv2D(64, (3, 3), strides=2, padding="same", activation='relu')(z)         # → (1, 4, 64)
    # z = BatchNormalization()(z)

    # Add bottleneck head
    # pooled = GlobalAveragePooling2D()(double_6)  # Shape: (None, 96)
    # latent = Dense(config.encoder.embedding_size, activation='relu', name='embedding')(pooled)  # e.g., 64

    # Instead of CNN, flattent the spatial tensor
    latent = Flatten(name="embedding")(z)

    return Model(inputs=x, outputs=latent, name="cnn_encoder")

# ------------------------------------------------------------------------
# Decoder Block
# ------------------------------------------------------------------------

def decoder_block(y, filters, stride=1):
    x = Conv2DTranspose(filters, (3, 3), strides=stride, padding="same")(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_decoder(config: BlazeGazeConfig):
    """
    Gradual decoder: expands latent vector → spatial tensor, then upscales to (128, 512, 3).
    """
    flat_dim = 1 * 4 * 16  # Must match encoder final tensor shape
    encoded_input = Input(shape=(config.encoder.embedding_size,), name='latent_vector')

    # Start from (1, 4, 64)
    # x = Dense(1 * 4 * 64, activation='relu', name='decoder_dense_expand')(encoded_input)
    # x = Reshape((1, 4, 64), name='decoder_reshape_initial')(x)
    # x = Reshape((1, 4, 64), name="decoder_reshape_initial")(encoded_input)
    x = Reshape((2, 8, 32), name="decoder_reshape_initial")(encoded_input)

    # Gradual upsampling: 7 steps to reach 128 x 512
    # Input (1x4x64)
    # x = decoder_block(x, 64, stride=2)  # → (2, 8)
    # x = decoder_block(x, 64, stride=2)  # → (4, 16)
    # x = decoder_block(x, 96, stride=2)  # → (8, 32)
    # x = decoder_block(x, 96, stride=2)  # → (16, 64)
    # x = decoder_block(x, 96, stride=2)  # → (32, 128)
    # x = decoder_block(x, 48, stride=2)  # → (64, 256)
    # x = decoder_block(x, 24, stride=2)  # → (128, 512)

    # Input (2x8x32)
    x = decoder_block(x, 96)            # 2x8 -> 2x8
    x = decoder_block(x, 96, stride=2)  # 2x8 -> 4x16
    x = decoder_block(x, 96)            # 4x16 -> 4x16
    x = decoder_block(x, 96, stride=2)  # 4x16 -> 8x32
    x = decoder_block(x, 48)            # 8x32 -> 8x32
    x = decoder_block(x, 48, stride=2)  # 8x32 -> 16x64
    x = decoder_block(x, 48)            # 16x64 -> 16x64
    x = decoder_block(x, 48, stride=2)  # 16x64 -> 32x128
    x = decoder_block(x, 24)            # 32x128 -> 32x128
    x = decoder_block(x, 24, stride=2)  # 32x128 -> 64x256
    x = decoder_block(x, 24)            # 64x256 -> 64x256
    x = decoder_block(x, 24, stride=2)  # 64x256 -> 128x512

    # Final RGB output
    output = Conv2D(3, (3, 3), padding="same", activation="sigmoid", name='decoder_output')(x)

    return tf.keras.Model(inputs=encoded_input, outputs=output, name="cnn_decoder")

# ------------------------------------------------------------------------
# Gaze Model
# ------------------------------------------------------------------------

def get_gaze_mlp(config):

    # handle the inputs
    print(config.encoder.embedding_size)
    cnn_input = Input(shape=[config.encoder.embedding_size], name="encoder_input")

    # Flatten instead of pooling to retain spatial information
    # flattened_output = Flatten(name="flattened_output")(cnn_input)

    # Handle optional inputs from config
    additional_inputs = []
    additional_tensors = []

    for input_cfg in config.gaze.inputs:
        input_name = input_cfg.name
        input_shape = input_cfg.input_shape
        input_tensor = Input(shape=input_shape, name=input_name)
        additional_inputs.append(input_tensor)
        additional_tensors.append(input_tensor)

    # Concatenate all features
    if additional_tensors:
        concat = Concatenate(name="feature_concat")([cnn_input] + additional_tensors)
    else:
        concat = cnn_input

    # MLP layers before computing the gaze vector
    # mlp1 = Dense(128, activation='relu')(concat)
    # mlp2 = Dense(64, activation='relu')(mlp1)
    # mlp3 = Dense(32, activation='relu')(mlp2)
    mlp1 = Dense(16, activation='relu')(concat)
    mlp2 = Dense(16, activation='relu')(mlp1)
    # mlp3 = Dense(32, activation='relu')(mlp2)

    output = Dense(2, activation='linear', name="gaze_output")(mlp2)
    
    # Build final model
    return Model(inputs=[cnn_input] + additional_inputs, outputs=output, name="gaze_mlp")

# ------------------------------------------------------------------------
# Production - Inference
# ------------------------------------------------------------------------

def build_full_inference_model(encoder, gaze_mlp, config):

    # Inputs
    image_input = Input(shape=config.encoder.input_shape, name='image')

    # Optional inputs
    additional_inputs = []
    additional_tensors = []
    for item in config.gaze.inputs:
        input_tensor = Input(shape=item.input_shape, name=item.name)
        additional_inputs.append(input_tensor)
        additional_tensors.append(input_tensor)

    # Pass image through encoder
    encoder_features = encoder(image_input, training=False)

    # Pass encoder features + additional inputs to gaze head
    gaze_inputs = [encoder_features] + additional_tensors
    gaze_output = gaze_mlp(gaze_inputs, training=False)

    # Final model with all inputs
    full_model = Model(
        inputs=[image_input] + additional_inputs,
        outputs=gaze_output,
        name="full_gaze_inference_model"
    )

    return full_model

# The BlazeGaze class is the main entry point for using the BlazeGaze model.
# It initializes the model based on the provided configuration and sets up the encoder, decoder, and gaze model.

class BlazeGaze():

    model: Model
    encoder: Model
    gaze_mlp: Model
    decoder: Optional[Model]

    def __init__(self, config: BlazeGazeConfig):
        self.config = config

        if self.config.weights_fp:
            self.load_assembled_model()
        else:
            self.load_specific_model()

        # Perform checks
        if self.decoder:
            assert self.encoder.input_shape == self.decoder.output_shape, f"Encoder input shape {self.encoder.input_shape} must match decoder output shape {self.decoder.output_shape}."
        
    def load_assembled_model(self):
        
        if isinstance(self.config.weights_fp, str):
            self.config.weights_fp = MODEL_WEIGHTS / self.config.weights_fp
        full_model = tf.keras.models.load_model(self.config.weights_fp, compile=False)
        self.model = full_model
        
        cnn_encoder = full_model.get_layer('cnn_encoder')  # or use exact last encoder layer name
        self.encoder = Model(
            inputs=cnn_encoder.inputs, 
            outputs=cnn_encoder.output, 
            name="cnn_encoder"
        )

        # Get encoder output tensor (which is input to the gaze model)
        gaze_mlp = full_model.get_layer('gaze_mlp')
        self.gaze_mlp = Model(
            inputs=gaze_mlp.inputs,
            outputs=gaze_mlp.output,
            name="gaze_mlp"
        )

        # If decoder exists, extract it
        if self.config.mode == 'autoencoder' and 'cnn_decoder' in full_model.layers:
            cnn_decoder = full_model.get_layer('cnn_decoder')
            self.decoder = Model(
                inputs=cnn_decoder.inputs,
                outputs=cnn_decoder.output,
                name="cnn_decoder"
            )
        else:
            self.decoder = None

    def load_specific_model(self):
        assert self.config.mode in ['autoencoder', 'maml'], "Unsupported mode. Use 'autoencoder' or 'maml'."
        
        self.encoder = get_cnn_encoder(self.config)
        self.gaze_mlp = get_gaze_mlp(self.config)
        if self.config.mode == 'autoencoder':
            self.decoder = get_decoder(self.config)
        else:
            self.decoder = None

        # Create input for the image
        image_input = Input(shape=self.config.encoder.input_shape, name='image')

        # Create Keras Input layers for extra gaze inputs
        additional_inputs = []
        for input_cfg in self.config.gaze.inputs:
            input_tensor = Input(shape=input_cfg.input_shape, name=input_cfg.name)
            additional_inputs.append(input_tensor)

        # Combine all inputs into a list
        model_inputs = [image_input] + additional_inputs

        # Propagate the image input through the encoder
        encoder_output = self.encoder(image_input)

        # The gaze model takes [encoder_output] + additional_inputs
        gaze_outputs = self.gaze_mlp([encoder_output] + additional_inputs)

        # Construct the full model outputs (encoder, gaze, and optionally decoder)
        model_outputs = [encoder_output, gaze_outputs] + ([self.decoder(encoder_output)] if self.decoder else [])
        self.model = Model(inputs=model_inputs, outputs=model_outputs, name="blazegaze_gaze")

        dummy_inputs = [tf.random.uniform((1, *self.config.encoder.input_shape))] + [
            tf.random.uniform((1, *inp.input_shape)) for inp in self.config.gaze.inputs
        ]
        self.model(dummy_inputs)

    def freeze_encoder(self):
        self.encoder.trainable = False
    
    def unfreeze_encoder(self):
        self.encoder.trainable = True

if __name__ == "__main__":

    TESTING = 'autoencoder'
    print("Testing mode:", TESTING)

    if TESTING == 'maml':
        # Test both models
        config = BlazeGazeConfig(
            # Mode
            mode='maml',
            # weights_fp='blazegaze_gazecapture.keras',
            # gaze=GazeConfig(
            #     inputs=[
            #         ModalityInput(name='head_vector', input_shape=(3,)),
            #         ModalityInput(name='face_origin_3d', input_shape=(3,)),
            #     ],
            #     output='2d'
            # )
        )
        model = BlazeGaze(config)
        model.model.summary()
        # model.encoder.summary()
        # model.gaze_mlp.summary()

        # Wrap in @tf.function
        # Critical for performance optimization
        @tf.function
        def infer_fn(*inputs):
            return model.model(inputs)

        inference_model = build_full_inference_model(
            encoder=model.encoder,
            gaze_mlp=model.gaze_mlp,
            config=config
        )
        inference_model.summary()

        for layer in inference_model.layers:
            print(layer.name, layer.trainable)

        # Test passing data through the model
        dummy_inputs = [
            tf.random.uniform((1, *config.encoder.input_shape))
        ] + [
            tf.random.uniform((1, *inp.input_shape)) for inp in config.gaze.inputs
        ]
        dummy_outputs = model.model(dummy_inputs)
        print("Dummy outputs shape:", [out.shape for out in dummy_outputs])

        # Warm-up
        infer_fn(*dummy_inputs)

        # Now run for N=100 and determine the time taken
        import time
        import tqdm
        N = 1000
        run_times = []
        for i in tqdm.tqdm(range(N), total=N):
            dummy_inputs = [
                tf.random.uniform((1, *config.encoder.input_shape))
            ] + [
                tf.random.uniform((1, *inp.input_shape)) for inp in config.gaze.inputs
            ]
            start_time = time.time()
            # dummy_outputs = model.model(dummy_inputs)
            _ = infer_fn(*dummy_inputs)
            end_time = time.time()
            run_times.append(end_time - start_time)

        print("Average time taken for inference:", np.mean(run_times))
        print("FPS:", 1 / np.mean(run_times))

    elif TESTING == 'autoencoder':
        # Test both models
        config = BlazeGazeConfig(
            # Mode
            mode='autoencoder',
            gaze=GazeConfig(
                inputs=[
                    ModalityInput(name='head_vector', input_shape=(3,)),
                    ModalityInput(name='face_origin_3d', input_shape=(3,)),
                ]
            )
        )
        model = BlazeGaze(config)
        model.model.summary()
        # model.encoder.summary()
        # model.decoder.summary()
        # model.gaze_mlp.summary()