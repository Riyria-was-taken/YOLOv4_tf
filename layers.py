import tensorflow as tf
import tensorflow_addons as tfa


class Mish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tfa.activations.mish(inputs)
