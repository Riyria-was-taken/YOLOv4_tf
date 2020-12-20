import tensorflow as tf

class YOLOv4Model:
    def __init__(self, image_size):
        input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        x = self.darknet53()(input)
        self.model = tf.keras.Model(input, x)

    def darknetResidualBlock(self, filters, repeats=1):
        def feed(x):
            x = tf.keras.layers.Conv2D(2 * filters, 3, strides=2)(x)
            for i in range(repeats):
                skip = x
                x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
                x = tf.keras.layers.Conv2D(2 * filters, 3, padding='same')(x)
                x = tf.keras.layers.Add()([skip, x])
            return x
        return feed

    def darknet53(self):
        def feed(x):
            x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
            x = self.darknetResidualBlock(32)(x)
            x = self.darknetResidualBlock(64, repeats=2)(x)
            x = self.darknetResidualBlock(128, repeats=8)(x)
            x = self.darknetResidualBlock(256, repeats=8)(x)
            x = self.darknetResidualBlock(512, repeats=4)(x)
            return x
        return feed

    def __call__(self, input):
        return self.model.predict(input)
