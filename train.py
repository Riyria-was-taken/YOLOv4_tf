from model import YOLOv4Model
import numpy as np
import tensorflow as tf

from img import read_img, draw_img
from pipeline import YOLOv4Pipeline
import utils

import math
import os


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
    truth_mask = tf.tensor_scatter_nd_update(truth_mask, indices_not_null, tf.ones(indices_not_null.shape[:-1], dtype=tf.float32))

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

anchor_sizes = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
scales = [1.2, 1.1, 1.05]


def train(file_root, annotations_file, batch_size, total_steps):
    model = YOLOv4Model()
    #model.load_weights("yolov4.weights")

    image_size = (608, 608)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), "little")

    pipeline = YOLOv4Pipeline(
        file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed
    )
    dataset = pipeline.dataset()

    var_list = model.model.trainable_weights
    iterator = iter(dataset)

    def loss():
        print("Data")
        input, gt_boxes = iterator.get_next()
        print("Infer")
        output = model(input)
        print("Loss")
        loss0 = calc_loss(0, gt_boxes, output[0])
        loss1 = calc_loss(1, gt_boxes, output[1])
        loss2 = calc_loss(2, gt_boxes, output[2])
        return loss0 + loss1 + loss2

    # needs verification
    optimizer=tf.keras.optimizers.Adam()
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = int(0.3 * total_steps)
    LR_INIT = 1e-3
    LR_END = 1e-6
    for i in range(total_steps):
        with tf.GradientTape() as tape:

            total_loss = loss()
            print("Optimize")
            gradients = tape.gradient(total_loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

            tf.print("=> TEST STEP %4d   lr: %.6f   loss: %4.2f" % (global_steps, optimizer.lr.numpy(), total_loss))

            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * LR_INIT
            else:
                lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

    return model





'''
model.model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tuple([lambda x, y : calc_loss(i, x, y) for i in range(3)]),
    loss_weights=[1.0, 1.0, 1.0]
)
model.model.fit(pipeline.dataset(), epochs=5, steps_per_epoch=10)
'''
