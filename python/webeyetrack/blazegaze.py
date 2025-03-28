from typing import Optional, Literal
from dataclasses import dataclass
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

@dataclass
class FourierTransformParams:
    trainable: bool = False
    mapping_size: int = 256
    scale: float = 10.0

# EMBEDDING_SIZE = 128
@dataclass
class BlazeGazeConfig:

    # Mode
    mode: Literal['autoencoder', 'gaze'] = 'autoencoder'
    weights_fp: Optional[str] = None

    # Encoder
    encoder_type: Literal['mlp', 'cnn'] = 'mlp'
    input_shape: tuple = (128, 512, 3)
    embedding_size: int = 512
    encoder_output: tuple = (8, 32, 96)

    # Decoder
    decoder_output: tuple = (128, 512, 3)

    # MLP
    gaze_output: Literal['2d', '3d'] = '3d'

    # Preprocessing
    fourier_transform: bool = False
    fourier_transform_params: FourierTransformParams = FourierTransformParams()

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

class FourierFeatureMapping(tf.keras.layers.Layer):
    def __init__(self, mapping_size=256, scale=10.0, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.mapping_size = mapping_size
        self.scale = scale
        self.trainable_B = trainable

    def build(self, input_shape):
        channels = input_shape[-1]
        # Trainable or fixed projection matrix (B)
        self.B = self.add_weight(
            shape=(channels, self.mapping_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
            trainable=self.trainable_B,  # Set to True if you want to learn B
            name="fourier_B"
        )

    def call(self, inputs):
        # inputs: [B, H, W, C]
        pi = tf.constant(np.pi, dtype=tf.float32)
        flat = tf.reshape(inputs, [-1, inputs.shape[-1]])  # [B*H*W, C]
        x_proj = tf.matmul(flat, self.B) * (2. * pi)
        x_proj = tf.reshape(x_proj, tf.concat([tf.shape(inputs)[:-1], [self.mapping_size]], axis=0))
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

class FourierFeatureVectorMapping(tf.keras.layers.Layer):
    def __init__(self, mapping_size=256, scale=10.0, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.mapping_size = mapping_size
        self.scale = scale
        self.trainable_B = trainable

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.B = self.add_weight(
            shape=(self.input_dim, self.mapping_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
            trainable=self.trainable_B,
            name="fourier_B_vector"
        )

    def call(self, inputs):
        # inputs: [batch_size, features]
        pi = tf.constant(np.pi, dtype=tf.float32)
        x_proj = tf.matmul(inputs, self.B) * (2. * pi)
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

def get_cnn_encoder(config: BlazeGazeConfig):
    x = Input(shape=config.input_shape)

    if config.fourier_transform:
        x_mapped = FourierFeatureMapping(
            mapping_size=config.fourier_transform_params.mapping_size,
            scale=config.fourier_transform_params.scale
        )(x)
    else:
        x_mapped = x

    # Feature extraction layers
    first_conv = Conv2D(24, (5,5), strides=2, padding="same", activation="relu")(x_mapped)
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

    return Model(inputs=x, outputs=double_6)

def get_mlp_encoder(config: BlazeGazeConfig):
    input_layer = Input(shape=config.input_shape)

    x = input_layer
    x = Flatten()(x)  # [B, H*W*C]

    if config.fourier_transform:
        x = FourierFeatureVectorMapping(
            mapping_size=config.fourier_transform_params.mapping_size,
            scale=config.fourier_transform_params.scale,
            trainable=config.fourier_transform_params.trainable
        )(x)

    # MLP layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(config.encoder_output), activation='relu')(x)
    x = tf.keras.layers.Reshape(config.encoder_output)(x)

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

    encoded_input = Input(shape=config.encoder_output)  # Adjust based on encoder output shape

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

    return Model(inputs=encoded_input, outputs=x)

# ------------------------------------------------------------------------
# Gaze Model
# ------------------------------------------------------------------------

def get_gaze_model(config):

    # handle the inputs
    x = Input(shape=config.encoder_output, name="encoder_input")

    # Additional 2D Convolutional Layer before pooling
    regularizer = tf.keras.regularizers.l2(1e-4)
    conv_layer = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(x)
    conv_layer = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(conv_layer)

    # Flatten instead of pooling to retain spatial information
    flattened_output = Flatten()(conv_layer)

    # Concatenate head rotation with flattened CNN features
    # concatenated_features = Concatenate()([flattened_output, head_rotation_input])

    # MLP layers before computing the gaze vector
    mlp = tf.keras.layers.Dense(128, activation='relu')(flattened_output)
    mlp = tf.keras.layers.Dense(64, activation='relu')(mlp)

    if config.gaze_output == '2d':
        # Output xy coordinate on the screen
        gaze_xy = tf.keras.layers.Dense(2, activation='linear', name="gaze_output")(mlp)
        return Model(inputs=x, outputs=gaze_xy)
    
    elif config.gaze_output == '3d':
        # Gaze vector output
        gaze_vector = tf.keras.layers.Dense(3, activation='linear', name="gaze_output")(mlp)
        # Ensure the output is normalized
        gaze_norm_vector = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="gaze_output_norm")(gaze_vector)
        return Model(inputs=x, outputs=gaze_norm_vector)

class BlazeGaze():

    model: Model

    def __init__(self, config: BlazeGazeConfig):
        self.config = config

        if self.config.mode == 'autoencoder':
            self.set_autoencoder()
        elif self.config.mode == 'gaze':
            self.set_gazemodel()
        else:
            raise ValueError("Invalid mode. Must be either 'autoencoder' or 'gaze'")

    def set_autoencoder(self):

        if self.config.encoder_type == 'cnn':
            self.encoder = get_cnn_encoder(self.config)
        elif self.config.encoder_type == 'mlp':
            self.encoder = get_mlp_encoder(self.config)
        else:
            raise ValueError("Invalid encoder type. Must be either 'cnn' or 'mlp'")
        self.decoder = get_decoder(self.config)
        self.model = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output), name="blazegaze_autoencoder")

        if self.config.weights_fp:
            self.model.load_weights(self.config.weights_fp)
        else:
            self.model([tf.random.uniform((1, *self.config.input_shape))])

    def set_gazemodel(self):
        if self.config.encoder_type == 'cnn':
            self.encoder = get_cnn_encoder(self.config)
        elif self.config.encoder_type == 'mlp':
            self.encoder = get_mlp_encoder(self.config)
        else:
            raise ValueError("Invalid encoder type. Must be either 'cnn' or 'mlp'")
        self.gaze_model = get_gaze_model(self.config)
        self.model = Model(inputs=self.encoder.input, outputs=self.gaze_model(self.encoder.output), name="blazegaze_gaze")

        if self.config.weights_fp:
            self.model.load_weights(self.config.weights_fp)
        else:
            self.model([tf.random.uniform((1, *self.config.input_shape))])

    def freeze_encoder(self):
        self.encoder.trainable = False
    
    def unfreeze_encoder(self):
        self.encoder.trainable = True

if __name__ == "__main__":

    # Test both models
    config = BlazeGazeConfig(
        # Mode
        mode='autoencoder',
        fourier_transform=True,
        fourier_transform_params=FourierTransformParams(trainable=True),
    )
    model = BlazeGaze(config)
    model.model.summary()


    # config = BlazeGazeConfig(
    #     # Mode
    #     mode='gaze'
    # )
    # model = BlazeGaze(config)
    # model.model.summary()
