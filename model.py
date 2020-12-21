import tensorflow as tf


class YOLOv4Model:
    def __init__(self, image_size):
        input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        x = self.darknet53()(input)
        self.model = tf.keras.Model(input, x)

    def darknetResidualBlock(self, filters, repeats=1):
        def feed(x):
            x = tf.keras.layers.Conv2D(2 * filters, 3, strides=2, activation="mish")(x)
            x = tf.keras.layers.Conv2D(filters, 1, padding="same", activation="mish")(x)
            route = x
            for i in range(repeats):
                skip = x
                x = tf.keras.layers.Conv2D(
                    filters, 1, padding="same", activation="mish"
                )(x)
                x = tf.keras.layers.Conv2D(
                    filters, 3, padding="same", activation="mish"
                )(x)
                x = tf.keras.layers.Add()([skip, x])
            x = tf.keras.layers.Conv2D(filters, 1, activation="mish")(x)
            x = tf.concat([x, route], axis=-1)
            x = tf.keras.layers.Conv2D(2 * filters, 1, activation="mish")(x)
            return x

        return feed

    def darknet53(self):
        def feed(x):
            x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="mish")(x)
            x = self.darknetResidualBlock(32)(x)
            x = self.darknetResidualBlock(64, repeats=2)(x)
            x = self.darknetResidualBlock(128, repeats=8)(x)
            r1 = x
            x = self.darknetResidualBlock(256, repeats=8)(x)
            r2 = x
            x = self.darknetResidualBlock(512, repeats=4)(x)
            x = tf.keras.layers.Conv2D(512, 1, activation="leaky")(x)
            x = tf.keras.layers.Conv2D(1024, 3, activation="leaky")(x)
            x = tf.keras.layers.Conv2D(512, 1, activation="leaky")(x)
            x = tf.concat(
                [
                    tf.nn.max_pool(x, ksize=13, padding="SAME", strides=1),
                    tf.nn.max_pool(x, ksize=9, padding="SAME", strides=1),
                    tf.nn.max_pool(x, ksize=5, padding="SAME", strides=1),
                    x,
                ],
                axis=-1,
            )
            x = tf.keras.layers.Conv2D(512, 1, activation="leaky")(x)
            x = tf.keras.layers.Conv2D(1024, 3, activation="leaky")(x)
            x = tf.keras.layers.Conv2D(512, 1, activation="leaky")(x)
            return r1, r2, x

        return feed

    def __call__(self, input):
        return self.model.predict(input)
