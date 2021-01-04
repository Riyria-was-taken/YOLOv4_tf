import tensorflow as tf
from layers import Mish


class YOLOv4Model:
    def __init__(self, classes_num=80, image_size=(608, 608)):
        self.classes_num = classes_num
        input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        output = self.CSPDarknet53WithSPP()(input)
        output = self.YOLOHead()(output)
        self.model = tf.keras.Model(input, output)

    def summary(self):
        self.model.summary()

    def darknetConv(
        self, filters, size, strides=1, batch_norm=True, activate=True, activation="leaky"
    ):
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

            if activate:
                if activation == "mish":
                    x = Mish()(x)
                elif activation == "leaky":
                    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

            return x

        return feed

    def darknetResidualBlock(self, filters, repeats=1, initial=False):
        def feed(x):
            filters2 = 2 * filters if initial else filters
            x = self.darknetConv(2 * filters, 3, strides=2, activation="mish")(x)
            route = self.darknetConv(filters2, 1, activation="mish")(x)
            x = self.darknetConv(filters2, 1, activation="mish")(x)
            for i in range(repeats):
                skip = x
                x = self.darknetConv(filters, 1, activation="mish")(x)
                x = self.darknetConv(filters2, 3, activation="mish")(x)
                x = tf.keras.layers.Add()([skip, x])
            x = self.darknetConv(filters2, 1, activation="mish")(x)
            x = tf.keras.layers.Concatenate()([x, route])
            x = self.darknetConv(2 * filters, 1, activation="mish")(x)
            return x

        return feed

    def CSPDarknet53WithSPP(self):
        def feed(x):
            x = self.darknetConv(32, 3, activation="mish")(x)
            x = self.darknetResidualBlock(32, initial=True)(x)
            x = self.darknetResidualBlock(64, repeats=2)(x)
            x = route_1 = self.darknetResidualBlock(128, repeats=8)(x)
            x = route_2 = self.darknetResidualBlock(256, repeats=8)(x)
            x = self.darknetResidualBlock(512, repeats=4)(x)
            x = self.darknetConv(512, 1)(x)
            x = self.darknetConv(1024, 3)(x)
            x = self.darknetConv(512, 1)(x)

            # SPP
            spp1 = tf.keras.layers.MaxPooling2D(pool_size=13, strides=1, padding="same")(x)
            spp2 = tf.keras.layers.MaxPooling2D(pool_size=9, strides=1, padding="same")(x)
            spp3 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=1, padding="same")(x)

            x = tf.keras.layers.Concatenate()([spp1, spp2, spp3, x])

            x = self.darknetConv(512, 1)(x)
            x = self.darknetConv(1024, 3)(x)
            x = self.darknetConv(512, 1)(x)
            return route_1, route_2, x

        return feed

    def yoloUpsampleConvBlock(self, filters):
        def feed(x, y):
            x = self.darknetConv(filters, 1)(x)
            x = tf.keras.layers.UpSampling2D()(x)
            y = self.darknetConv(filters, 1)(y)
            x = tf.keras.layers.Concatenate()([y, x])

            x = self.darknetConv(filters, 1)(x)
            x = self.darknetConv(2 * filters, 3)(x)
            x = self.darknetConv(filters, 1)(x)
            x = self.darknetConv(2 * filters, 3)(x)
            x = self.darknetConv(filters, 1)(x)

            return x

        return feed

    def yoloDownsampleConvBlock(self, filters):
        def feed(x, y):
            x = self.darknetConv(filters, 3, strides=2)(x)
            x = tf.keras.layers.Concatenate()([x, y])

            x = self.darknetConv(filters, 1)(x)
            x = self.darknetConv(2 * filters, 3)(x)
            x = self.darknetConv(filters, 1)(x)
            x = self.darknetConv(2 * filters, 3)(x)
            x = self.darknetConv(filters, 1)(x)

            return x

        return feed

    def yoloBboxConvBlock(self, filters):
        def feed(x):
            x = self.darknetConv(filters, 3)(x)
            x = self.darknetConv(3 * (self.classes_num + 5), 1, activate=False, batch_norm=False)(x)

            return x

        return feed

    def YOLOHead(self):
        def feed(x):
            route_1, route_2, route = x
            x = route_2 = self.yoloUpsampleConvBlock(256)(route, route_2)
            x = route_1 = self.yoloUpsampleConvBlock(128)(x, route_1)
            small_bbox = self.yoloBboxConvBlock(256)(x)
            x = self.yoloDownsampleConvBlock(256)(route_1, route_2)
            medium_bbox = self.yoloBboxConvBlock(512)(x)
            x = self.yoloDownsampleConvBlock(512)(x, route)
            large_bbox = self.yoloBboxConvBlock(1024)(x)

            return small_bbox, medium_bbox, large_bbox

        return feed

    def __call__(self, input):
        return self.model.predict(input)
