from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, DepthwiseConv2D, Conv2D, MaxPool2D, Add, Activation, Flatten

class HeadWrapper(Layer):
    def __init__(self, last_dimension, **kwargs):
        super(HeadWrapper, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self):
        config = super(HeadWrapper, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, xs):
        last_dimension = self.last_dimension
        batch_size = tf.shape(xs[0])[0]
        outputs = []
        for conv_layer in xs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
        return tf.concat(outputs, axis=1)

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

def get_backbone(input_shape=(128, 128, 3)):
    x = Input(shape=input_shape)

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

    return Model(inputs=x, outputs=double_6)

def get_gaze_model(input_shape=(128, 128, 3)):
    x = Input(shape=input_shape)

    # BlazeGaze backbone
    backbone = get_backbone(input_shape=input_shape)
    double_6 = backbone(x)

    # Additional 2D Convolutional Layer before pooling
    regularizer = tf.keras.regularizers.l2(1e-4)
    conv_layer = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(double_6)
    conv_layer = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizer)(conv_layer)

    # Flatten instead of pooling to retain spatial information
    flattened_output = Flatten()(conv_layer)

    # MLP layers before computing the gaze vector
    mlp = tf.keras.layers.Dense(128, activation='relu')(flattened_output)
    mlp = tf.keras.layers.Dense(64, activation='relu')(mlp)
    mlp_output = tf.keras.layers.Dense(3, activation='linear', name="mlp_output")(mlp)

    # Add epsilon before normalization to avoid division by zero
    epsilon = 1e-8
    mlp_output = tf.keras.layers.Lambda(lambda x: x + epsilon)(mlp_output)

    # Normalize the gaze vector
    gaze_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1), name="gaze_output")(mlp_output)

    return Model(inputs=x, outputs=gaze_output), backbone

def init_model(model):
    model(tf.random.uniform((1, 128, 128, 3)))

class BlazeGaze():

    def __init__(self, weights_fp: Optional[str] = None):
        self.tf_model, self.tf_model_backbone = get_gaze_model()

        if weights_fp:
            self.tf_model.load_weights(weights_fp)
        else:
            init_model(self.tf_model)

    def freeze_backbone(self):
        self.tf_model_backbone.trainable = False

    def unfreeze_backbone(self):
        self.tf_model_backbone.trainable = True

if __name__ == "__main__":
    model = get_gaze_model()
    init_model(model)
    model.summary()
