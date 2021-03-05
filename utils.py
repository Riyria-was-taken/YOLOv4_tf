import numpy as np
import tensorflow as tf
import numpy as np

def xywh_to_ltrb(boxes):
    boxes = tf.convert_to_tensor(boxes)
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    return tf.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

def calc_ious(boxes1, boxes2):
    ltrb1 = xywh_to_ltrb(boxes1)
    ltrb2 = xywh_to_ltrb(boxes2)

    il = tf.math.maximum(ltrb1[..., 0], ltrb2[..., 0])
    it = tf.math.maximum(ltrb1[..., 1], ltrb2[..., 1])
    ir = tf.math.minimum(ltrb1[..., 2], ltrb2[..., 2])
    ib = tf.math.minimum(ltrb1[..., 3], ltrb2[..., 3])
    I = tf.math.maximum(0, ir - il) * tf.math.maximum(0, ib - it)

    A1 = (ltrb1[..., 2] - ltrb1[..., 0]) * (ltrb1[..., 3] - ltrb1[..., 1])
    A2 = (ltrb2[..., 2] - ltrb2[..., 0]) * (ltrb2[..., 3] - ltrb2[..., 1])
    U = A1 + A2 - I

    return I / U

def calc_gious(boxes1, boxes2):
    ltrb1 = xywh_to_ltrb(boxes1)
    ltrb2 = xywh_to_ltrb(boxes2)

    il = tf.math.maximum(ltrb1[..., 0], ltrb2[..., 0])
    it = tf.math.maximum(ltrb1[..., 1], ltrb2[..., 1])
    ir = tf.math.minimum(ltrb1[..., 2], ltrb2[..., 2])
    ib = tf.math.minimum(ltrb1[..., 3], ltrb2[..., 3])
    I = tf.math.maximum(0, ir - il) * tf.math.maximum(0, ib - it)

    A1 = (ltrb1[..., 2] - ltrb1[..., 0]) * (ltrb1[..., 3] - ltrb1[..., 1])
    A2 = (ltrb2[..., 2] - ltrb2[..., 0]) * (ltrb2[..., 3] - ltrb2[..., 1])
    U = A1 + A2 - I

    cl = tf.math.minimum(ltrb1[..., 0], ltrb2[..., 0])
    ct = tf.math.minimum(ltrb1[..., 1], ltrb2[..., 1])
    cr = tf.math.minimum(ltrb1[..., 2], ltrb2[..., 2])
    cb = tf.math.minimum(ltrb1[..., 3], ltrb2[..., 3])
    C = (cr - cl) * (cb - ct)

    return I / U - (C - U) / C
