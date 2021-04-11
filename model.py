import tensorflow as tf
import numpy as np
from layers import Mish, ScaledRandomUniform
import utils



anchor_sizes = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
scales = [1.2, 1.1, 1.05]

def calc_loss(layer_id, gt, preds, debug=False):
    gt_boxes = gt[..., : 4]
    gt_labels = tf.cast(gt[..., 4], tf.int32)
    gt_mask = tf.where(gt_labels == -1, 0.0, 1.0)
    layer_xywh, layer_obj, layer_cls = utils.decode_layer(preds, layer_id)
    cls_count = layer_cls.shape[-1]

    s = tf.shape(preds) # (batch, x, y, ratio * stuff)
    batch_size = s[0]
    gw = s[1]
    gh = s[2]
    stride_x = 1 / gw
    stride_y = 1 / gh
    d = s[3]
    truth_mask = tf.zeros((batch_size, gw, gh, 3))

    box_loss = 0.0
    cls_loss = 0.0

    ix = tf.cast(tf.math.floor(tf.cast(gw, tf.float32) * gt_boxes[..., 0]), tf.int32)
    iy = tf.cast(tf.math.floor(tf.cast(gh, tf.float32) * gt_boxes[..., 1]), tf.int32)
    ix = tf.clip_by_value(ix, 0, gw - 1)
    iy = tf.clip_by_value(iy, 0, gh - 1)

    box_shape = tf.shape(gt_labels)
    zeros = tf.zeros_like(gt_labels, dtype=tf.float32)
    gt_shift = tf.stack([zeros, zeros, gt_boxes[..., 2], gt_boxes[..., 3]], axis=-1)
    gt_shift = tf.stack([gt_shift, gt_shift, gt_shift], axis=1)

    anchors_ws = [tf.cast(tf.fill(box_shape, anchor_sizes[layer_id][ir][0]), dtype=tf.float32) / 608.0 for ir in range(3)]
    anchors_hs = [tf.cast(tf.fill(box_shape, anchor_sizes[layer_id][ir][1]), dtype=tf.float32) / 608.0 for ir in range(3)]
    anchors = tf.stack([tf.stack([zeros, zeros, anchors_ws[ir], anchors_hs[ir]], axis=-1) for ir in range(3)], axis=1)

    ious = utils.calc_ious(gt_shift, anchors)
    ious_argmax = tf.cast(tf.argmax(ious, axis=1), dtype=tf.int32)
    batch_idx = tf.tile(tf.range(batch_size)[ : , tf.newaxis], [1, box_shape[-1]])

    indices = tf.stack([batch_idx, iy, ix, ious_argmax], axis=-1)
    pred_boxes = tf.gather_nd(layer_xywh, indices)
    box_loss = tf.math.reduce_sum(gt_mask * (1.0 - utils.calc_gious(pred_boxes, gt_boxes)))

    cls_one_hot = tf.one_hot(gt_labels, cls_count)
    pred_cls = tf.gather_nd(layer_cls, indices)
    cls_diffs = tf.math.reduce_sum(tf.math.square(pred_cls - cls_one_hot), axis=-1)
    cls_loss = tf.math.reduce_sum(gt_mask * cls_diffs)

    indices_not_null = tf.gather_nd(indices, tf.where(gt_labels != -1))
    truth_mask = tf.tensor_scatter_nd_update(truth_mask, indices_not_null, tf.ones_like(indices_not_null, dtype=tf.float32)[:,0])

    # TODO: add iou masking for noobj loss
    obj_loss = tf.math.reduce_sum(tf.math.square(truth_mask - layer_obj))

#    for ir in range(3):
#        data = layer[..., (d // 3) * ir : (d // 3) * (ir + 1)]
#        pred_boxes = data[0, ..., : 4]
#        objectness = data[0, ..., 4]
#
#        best_ious = tf.zeros([1, gw, gh])
#        for gt_box in gt_boxes[0]:
#            gt_box_exp = tf.tile(tf.reshape(gt_box, (1, 1, 1, 4)), [1, gw, gh, 1])
#            ious = utils.calc_ious(pred_boxes, gt_box_exp)
#            best_ious = tf.math.maximum(best_ious, ious)
#        iou_mask = tf.cast(best_ious < 0.7, tf.float32)
#
#        obj_loss += tf.math.reduce_sum(tf.math.square(1 - objectness) * truth_mask[..., ir])
#        obj_loss += tf.math.reduce_sum(tf.math.square(objectness) * inv_truth_mask[..., ir] * iou_mask)

    return (box_loss + obj_loss + cls_loss) / tf.cast(batch_size, dtype=tf.float32)





class YOLOv4Training(tf.keras.Model):
    def __init__(self, input, output):
        super(YOLOv4Training, self).__init__(input, output)
        self.lr_init = 1e-3
        self.lr_end = 1e-6
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.lr_tracker = tf.keras.metrics.Mean(name="lr")

    def fit(self, dataset, **kwargs):
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int32)
        self.total_steps = kwargs['epochs'] + kwargs['steps_per_epoch']
        self.warmup_steps = int(0.3 * self.total_steps)
        super(YOLOv4Training, self).fit(dataset, **kwargs)

    def train_step(self, data):

        lr_warmup = self.global_steps / self.warmup_steps * self.lr_init
        lr_main = self.lr_end + 0.5 * (self.lr_init - self.lr_end) * (
            (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
        )
        lr = tf.cond(self.global_steps < self.warmup_steps, lambda: lr_warmup, lambda: lr_main)
        self.optimizer.lr.assign(tf.cast(lr, dtype=tf.float32))

        #print("Data")
        input, gt_boxes = data

        with tf.GradientTape() as tape:

            #print("Infer")
            output = self(input, training=True)

            #print("Loss")
            loss0 = calc_loss(0, gt_boxes, output[0])
            loss1 = calc_loss(1, gt_boxes, output[1])
            loss2 = calc_loss(2, gt_boxes, output[2])
            total_loss = loss0 + loss1 + loss2
            #total_loss = loss0
            #print(total_loss)

            #print("Optimize")
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.lr_tracker.update_state(self.optimizer.lr)

        #tf.print("=> TEST STEP %4d   lr: %.6f   loss: %4.2f" % (global_steps, optimizer.lr.numpy(), total_loss))

        self.global_steps.assign_add(1)
        #if self.global_steps < self.warmup_steps:
        #    lr = self.global_steps / self.warmup_steps * self.lr_init
        #else:
        #    lr = self.lr_end + 0.5 * (self.lr_init - self.lr_end) * (
        #        (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
        #    )
        #self.optimizer.lr.assign(lr.numpy())

        return {"loss" : self.loss_tracker.result(), "lr" : self.lr_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.lr_tracker]


class YOLOv4Model:
    def __init__(self, classes_num=80, image_size=(608, 608)):
        self.classes_num = classes_num
        self.image_size = (image_size[0], image_size[1], 3)
        input = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
        output = self.CSPDarknet53WithSPP()(input)
        output = self.YOLOHead()(output)
        self.model = YOLOv4Training(input, output)

    def summary(self):
        self.model.summary()


    def load_weights(self, weights_file):
        if weights_file.endswith(".h5"):
            self._load_weights_tf(weights_file)
        else:
            self._load_weights_yolo(weights_file)

    def save_weights(self, weights_file):
        self._save_weights_tf(weights_file)


    def _load_weights_yolo(self, weights_file):
        with open(weights_file, "rb") as f:
            major, minor, revision = np.fromfile(f, dtype=np.int32, count=3)
            if (major * 10 + minor) >= 2:
                seen = np.fromfile(f, dtype=np.int64, count=1)
            else:
                seen = np.fromfile(f, dtype=np.int32, count=1)
            j = 0
            for i in range(110):
                conv_layer_name = "conv2d_%d" % i if i > 0 else "conv2d"
                bn_layer_name = "batch_normalization_%d" % j if j > 0 else "batch_normalization"

                conv_layer = self.model.get_layer(conv_layer_name)
                in_dim = conv_layer.input_shape[-1]
                filters = conv_layer.filters
                size = conv_layer.kernel_size[0]

                if i not in [93, 101, 109]:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(f, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = self.model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(f, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(f, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in [93, 101, 109]:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(f.read()) == 0, "failed to read all data"

    def _save_weights_tf(self, weights_file):
        self.model.save_weights(weights_file, save_format='h5')

    def _load_weights_tf(self, weights_file):
        self.model.load_weights(weights_file)


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
                kernel_initializer=ScaledRandomUniform(
                    scale=tf.sqrt(2 / (size * size * self.image_size[2])), minval=-0.01, maxval=0.01
                ),
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            )(x)

            if batch_norm:
                x = tf.keras.layers.BatchNormalization(moving_variance_initializer="zeros", momentum = 0.5)(x)

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


    def predict(self, input):
        return self.model.predict(input)

    def __call__(self, input, training=True):
        return self.model(input, training=training)
