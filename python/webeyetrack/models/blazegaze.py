import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, DepthwiseConv2D, Conv2D, MaxPool2D, Add, Activation

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

def get_gaze_model(input_shape=(128, 128, 3)):
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

    # Output layer for gaze estimation (3D gaze vector)
    gaze_output = Conv2D(3, (1, 1), activation="linear", name="gaze_output")(double_6)
    # gaze_output = Conv2D(3, (1, 1), activation="linear", name="gaze_output")(double_6)

    # Normalize the gaze vector
    gaze_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(gaze_output)

    # Check for NaN values
    gaze_output = tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))(gaze_output)

    # Flatten output
    gaze_output = tf.keras.layers.GlobalAveragePooling2D()(gaze_output)

    return Model(inputs=x, outputs=gaze_output)

def init_model(model):
    model(tf.random.uniform((1, 128, 128, 3)))

if __name__ == "__main__":
    model = get_gaze_model()
    init_model(model)
    model.summary()
