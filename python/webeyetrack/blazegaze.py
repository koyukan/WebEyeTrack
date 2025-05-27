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
    BatchNormalization
)

# Optional: Enable XLA for optimization
tf.config.optimizer.set_jit(True)

from .constants import MODEL_WEIGHTS

@dataclass
class FourierTransformParams:
    trainable: bool = False
    mapping_size: int = 256
    scale: float = 10.0

@dataclass
class ModalityInput:
    name: str
    input_shape: tuple

@dataclass
class EncoderConfig:
    weights_fp: Optional[str] = None
    type: Literal['mlp', 'cnn'] = 'cnn'
    input_shape: tuple = (128, 512, 3)
    embedding_size: int = 512
    output_shape: tuple = (8, 32, 96)

@dataclass
class DecoderConfig:
    output_shape: tuple = (128, 512, 3)

@dataclass
class GazeConfig:
    inputs: List[ModalityInput] = field(default_factory=lambda: [])
    output: Literal['2d', '3d'] = '3d'

@dataclass
class BlazeGazeConfig:

    # Mode
    mode: Literal['autoencoder', 'gaze'] = 'autoencoder'
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

# class FourierFeatureMapping(tf.keras.layers.Layer):
#     def __init__(self, mapping_size=256, scale=10.0, trainable=False, **kwargs):
#         super().__init__(**kwargs)
#         self.mapping_size = mapping_size
#         self.scale = scale
#         self.trainable_B = trainable

#     def build(self, input_shape):
#         channels = input_shape[-1]
#         # Trainable or fixed projection matrix (B)
#         self.B = self.add_weight(
#             shape=(channels, self.mapping_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
#             trainable=self.trainable_B,  # Set to True if you want to learn B
#             name="fourier_B"
#         )

#     def call(self, inputs):
#         # inputs: [B, H, W, C]
#         pi = tf.constant(np.pi, dtype=tf.float32)
#         flat = tf.reshape(inputs, [-1, inputs.shape[-1]])  # [B*H*W, C]
#         x_proj = tf.matmul(flat, self.B) * (2. * pi)
#         x_proj = tf.reshape(x_proj, tf.concat([tf.shape(inputs)[:-1], [self.mapping_size]], axis=0))
#         return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

# class FourierFeatureVectorMapping(tf.keras.layers.Layer):
#     def __init__(self, mapping_size=256, scale=10.0, trainable=False, **kwargs):
#         super().__init__(**kwargs)
#         self.mapping_size = mapping_size
#         self.scale = scale
#         self.trainable_B = trainable

#     def build(self, input_shape):
#         self.input_dim = input_shape[-1]
#         self.B = self.add_weight(
#             shape=(self.input_dim, self.mapping_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
#             trainable=self.trainable_B,
#             name="fourier_B_vector"
#         )

#     def call(self, inputs):
#         # inputs: [batch_size, features]
#         pi = tf.constant(np.pi, dtype=tf.float32)
#         x_proj = tf.matmul(inputs, self.B) * (2. * pi)
#         return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

def get_cnn_encoder(config: BlazeGazeConfig):
    x = Input(shape=config.encoder.input_shape)

    # if config.fourier_transform:
    #     x_mapped = FourierFeatureMapping(
    #         mapping_size=config.fourier_transform_params.mapping_size,
    #         scale=config.fourier_transform_params.scale
    #     )(x)
    # else:
    #     x_mapped = x

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

    return Model(inputs=x, outputs=double_6, name="cnn_encoder")


def get_mlp_encoder(config: BlazeGazeConfig):
    input_layer = Input(shape=config.encoder.input_shape)

    x = input_layer
    x = Flatten()(x)  # [B, H*W*C]

    # if config.fourier_transform:
    #     x = FourierFeatureVectorMapping(
    #         mapping_size=config.fourier_transform_params.mapping_size,
    #         scale=config.fourier_transform_params.scale,
    #         trainable=config.fourier_transform_params.trainable
    #     )(x)

    # MLP layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(config.encoder.output_shape), activation='relu')(x)
    x = tf.keras.layers.Reshape(config.encoder.output_shape)(x)

    return Model(inputs=input_layer, outputs=x, name="mlp_encoder")

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
    CNN-based decoder that reconstructs image from encoded features.
    Mirrors the encoder structure and upscales back to (H, W, 3).
    """

    encoded_input = Input(shape=config.encoder.output_shape)  # Adjust based on encoder output shape

    x = decoder_block(encoded_input, 96)         # 8x32 → 8x32
    x = decoder_block(x, 96)                     # 8x32 → 8x32
    x = decoder_block(x, 96, stride=2)           # 8x32 → 16x64
    x = decoder_block(x, 96)                     # 16x64 → 16x64
    x = decoder_block(x, 96)                     # 16x64 → 16x64
    x = decoder_block(x, 48, stride=2)           # 16x64 → 32x128
    x = decoder_block(x, 48)                     # 32x128 → 32x128
    x = decoder_block(x, 24, stride=2)           # 32x128 → 64x256
    x = decoder_block(x, 24)                     # 64x256 → 64x256
    x = decoder_block(x, 24, stride=2)           # 64x256 → 128x512

    # Final RGB reconstruction
    x = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)  # Output in [0, 1]

    return Model(inputs=encoded_input, outputs=x, name="cnn_decoder")

# ------------------------------------------------------------------------
# Gaze Model
# ------------------------------------------------------------------------

def get_gaze_model(config):

    # handle the inputs
    cnn_input = Input(shape=config.encoder.output_shape, name="encoder_input")

    # Additional 2D Convolutional Layer before pooling
    regularizer = tf.keras.regularizers.l2(1e-4)
    conv_layer = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(cnn_input)
    conv_layer = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(conv_layer)

    # Flatten instead of pooling to retain spatial information
    flattened_output = Flatten()(conv_layer)

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
        concat = Concatenate(name="feature_concat")([flattened_output] + additional_tensors)
    else:
        concat = flattened_output

    # MLP layers before computing the gaze vector
    mlp = tf.keras.layers.Dense(128, activation='relu')(concat)
    mlp = tf.keras.layers.Dense(64, activation='relu')(mlp)

    if config.gaze.output == '2d':
        output = tf.keras.layers.Dense(2, activation='linear', name="gaze_output")(mlp)
    
    elif config.gaze.output == '3d':
        gaze_vector = tf.keras.layers.Dense(3, activation='linear', name="gaze_output")(mlp)
        output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="gaze_output_norm")(gaze_vector)
    
    # Build final model
    return Model(inputs=[cnn_input] + additional_inputs, outputs=output, name="gaze_head")

# ------------------------------------------------------------------------
# Production - Inference
# ------------------------------------------------------------------------

def build_full_inference_model(encoder, gaze_head, config):

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
    gaze_output = gaze_head(gaze_inputs, training=False)

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
    decoder: Optional[Model]
    gaze_model: Optional[Model]

    def __init__(self, config: BlazeGazeConfig):
        self.config = config

        if self.config.weights_fp:
            self.load_entire_model()
        elif self.config.mode == 'autoencoder':
            self.set_autoencoder()
        elif self.config.mode == 'gaze':
            self.set_gazemodel()
        else:
            raise ValueError("Invalid mode. Must be either 'autoencoder' or 'gaze'")
        
    def load_entire_model(self):
        
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
        gaze_head = full_model.get_layer('gaze_head')
        self.gaze_model = Model(
            inputs=gaze_head.inputs,
            outputs=gaze_head.output,
            name="gaze_head"
        )

    def set_autoencoder(self):

        if self.config.encoder.type == 'cnn':
            self.encoder = get_cnn_encoder(self.config)
        elif self.config.encoder.type == 'mlp':
            self.encoder = get_mlp_encoder(self.config)
        else:
            raise ValueError("Invalid encoder type. Must be either 'cnn' or 'mlp'")
        self.decoder = get_decoder(self.config)
        self.model = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output), name="blazegaze_autoencoder")

        if self.config.weights_fp:
            self.model.load_weights(self.config.weights_fp)
        else:
            self.model([tf.random.uniform((1, *self.config.encoder.input_shape))])

    def set_gazemodel(self):
        if self.config.encoder.type == 'cnn':
            self.encoder = get_cnn_encoder(self.config)
        elif self.config.encoder.type == 'mlp':
            self.encoder = get_mlp_encoder(self.config)
        else:
            raise ValueError("Invalid encoder type. Must be either 'cnn' or 'mlp'")

        self.gaze_model = get_gaze_model(self.config)

        # Create Keras Input layers for extra gaze inputs
        additional_inputs = []
        for input_cfg in self.config.gaze.inputs:
            input_tensor = Input(shape=input_cfg.input_shape, name=input_cfg.name)
            additional_inputs.append(input_tensor)

        # The gaze model takes [encoder_output] + additional_inputs
        model_inputs = [self.encoder.input] + additional_inputs
        gaze_outputs = self.gaze_model([self.encoder.output] + additional_inputs)

        self.model = Model(inputs=model_inputs, outputs=gaze_outputs, name="blazegaze_gaze")

        # Load weights if provided
        if self.config.weights_fp:
            self.model.load_weights(self.config.weights_fp)
        else:
            dummy_inputs = [tf.random.uniform((1, *self.config.encoder.input_shape))] + [
                tf.random.uniform((1, *inp.input_shape)) for inp in self.config.gaze.inputs
            ]
            self.model(dummy_inputs)

    def freeze_encoder(self):
        self.encoder.trainable = False
    
    def unfreeze_encoder(self):
        self.encoder.trainable = True

if __name__ == "__main__":

    # Test both models
    config = BlazeGazeConfig(
        # Mode
        mode='gaze',
        weights_fp='blazegaze_gazecapture.keras',
        gaze=GazeConfig(
            inputs=[
                ModalityInput(name='head_vector', input_shape=(3,)),
                ModalityInput(name='face_origin_3d', input_shape=(3,)),
            ],
            output='2d'
        )
    )
    model = BlazeGaze(config)
    model.model.summary()

    # Wrap in @tf.function
    @tf.function
    def infer_fn(*inputs):
        return model.model(inputs)

    inference_model = build_full_inference_model(
        encoder=model.encoder,
        gaze_head=model.gaze_model,
        config=config
    )
    inference_model.summary()

    for layer in inference_model.layers:
        print(layer.name, layer.trainable)

    # config = BlazeGazeConfig(
    #     # Mode
    #     mode='gaze'
    # )
    # model = BlazeGaze(config)
    # model.model.summary()

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
    N = 100
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
