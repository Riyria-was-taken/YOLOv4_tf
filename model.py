import tensorflow as tf
from layers import Mish


class YOLOv4Model:
    def __init__(self, image_size):
        input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        output = self.CSPDarknet53WithSPP()(input)
        self.model = tf.keras.Model(input, output)

    def summary(self):
        self.model.summary()

    def darknetConv(self, filters, size, strides=1, batch_norm=True, activation="mish"):
        def feed(x):
            if strides == 1:
                padding = "same"
            else:
                x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
                padding = "valid"

            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=size,
                strides=strides,
                padding=padding,
                use_bias=not batch_norm,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            )(x)

            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)

            if activation == "mish":
                x = Mish()(x)
            elif activation == "leaky":
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            return x

        return feed

    def darknetResidualBlock(self, filters, repeats=1):
        def feed(x):
            x = self.darknetConv(2 * filters, 3, strides=2)(x)
            x = self.darknetConv(filters, 1)(x)
            route = x
            for i in range(repeats):
                skip = x
                x = self.darknetConv(filters, 1)(x)
                x = self.darknetConv(filters, 3)(x)
                x = tf.keras.layers.Add()([skip, x])
            x = self.darknetConv(filters, 1)(x)
            x = tf.concat([x, route], axis=-1)
            x = self.darknetConv(2 * filters, 1)(x)
            return x

        return feed

    def CSPDarknet53WithSPP(self):
        def feed(x):
            x = self.darknetConv(32, 3)(x)
            x = self.darknetResidualBlock(32)(x)
            x = self.darknetResidualBlock(64, repeats=2)(x)
            x = r1 = self.darknetResidualBlock(128, repeats=8)(x)
            x = r2 = self.darknetResidualBlock(256, repeats=8)(x)
            x = self.darknetResidualBlock(512, repeats=4)(x)
            x = self.darknetConv(512, 1, activation="leaky")(x)
            x = self.darknetConv(1024, 3, activation="leaky")(x)
            x = self.darknetConv(512, 1, activation="leaky")(x)

            # SPP
            spp1 = tf.keras.layers.MaxPooling2D(pool_size=13, strides=1, padding="same")(x)
            spp2 = tf.keras.layers.MaxPooling2D(pool_size=9, strides=1, padding="same")(x)
            spp3 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=1, padding="same")(x)
            x = tf.keras.layers.Concatenate()([spp1, spp2, spp3, x])

            x = self.darknetConv(512, 1, activation="leaky")(x)
            x = self.darknetConv(1024, 3, activation="leaky")(x)
            x = self.darknetConv(512, 1, activation="leaky")(x)
            return r1, r2, x

        return feed

    def __call__(self, input):
        return self.model.predict(input)
