from model import YOLOv4Model
import numpy as np
import tensorflow as tf

from img import read_img, draw_img

import math

import utils


def calc_loss(gt_boxes, gt_labels, preds):
    tot_box_loss = 0
    tot_obj_loss = 0
    tot_cls_loss = 0
    for i, preds in enumerate(output):
        s = preds.shape
        gw, gh = s[1:3]
        stride_x = 1 / gw
        stride_y = 1 / gh
        d = s[3]
        truth_mask = np.zeros([1, gw, gh, 3], dtype=np.float32)

        box_loss = 0
        cls_loss = 0
        for gt_box, gt_cls in zip(gt_boxes, gt_labels):
            ix = math.floor(gw * gt_box[0])
            iy = math.floor(gh * gt_box[1])

            best_ar_iou = 0
            best_ar = 0
            gt_shift = (0, 0, gt_box[2], gt_box[3])

            good = []
            for ir in range(3):
                w, h = anchor_sizes[i][ir]
                w /= 608
                h /= 608
                iou = utils.calc_ious(gt_shift, (0, 0, w, h))
                if iou > best_ar_iou:
                    best_ar_iou = iou
                    best_ar = ir
                if iou > 0.213:
                    good.append(ir)
            if len(good) == 0:
                good.append(best_ar)

            for ir in good:
                truth_mask[0, iy, ix, ir] = 1

                data = preds[0, iy, ix, (d // 3) * ir : (d // 3) * (ir + 1)]

                dx, dy, dw, dh = data[:4]
                x = (utils.sigmoid(dx) * scales[i] - 0.5 * (scales[i] - 1) + ix) * stride_x
                y = (utils.sigmoid(dy) * scales[i] - 0.5 * (scales[i] - 1) + iy) * stride_y

                w, h = anchor_sizes[i][ir]
                w /= 608
                h /= 608
                w *= math.exp(dw)
                h *= math.exp(dh)
                pred_box = [x, y, w, h]

                box_loss += 1 - utils.calc_gious(gt_box[:4], pred_box)
                cls_preds = [utils.sigmoid(x) for x in data[5:]]
                for ic in range(len(cls_preds)):
                    pred = cls_preds[ic]
                    truth = 1 if gt_cls == ic else 0
                    cls_loss += (pred - truth)**2

        inv_truth_mask = 1 - truth_mask
        obj_loss = 0
        ix = tf.cast(tf.tile(tf.expand_dims(tf.range(gw), axis=0), [gw, 1]), tf.float32)
        iy = tf.cast(tf.tile(tf.expand_dims(tf.range(gw), axis=1), [1, gh]), tf.float32)
        for ir in range(3):
            data = preds[:, :, :, (d // 3) * ir : (d // 3) * (ir + 1)]
            dx = data[:, :, :, 0]
            dy = data[:, :, :, 1]
            dw = data[:, :, :, 2]
            dh = data[:, :, :, 3]
            x = (utils.sigmoid(dx) * scales[i] - 0.5 * (scales[i] - 1) + ix) * stride_x
            y = (utils.sigmoid(dy) * scales[i] - 0.5 * (scales[i] - 1) + iy) * stride_y
            w = tf.math.exp(dw) * anchor_sizes[i][ir][0] / 608
            h = tf.math.exp(dh) * anchor_sizes[i][ir][1] / 608
            pred_boxes = tf.stack([x, y, w, h], axis=3)
            objectness = utils.sigmoid(data[:, :, :, 4])

            best_ious = tf.zeros([1, gw, gh])
            for gt_box in gt_boxes:
                gt_box_exp = tf.tile(tf.reshape(gt_box[:4], (1, 1, 1, 4)), [1, gw, gh, 1])
                ious = utils.calc_ious(pred_boxes, gt_box_exp)
                best_ious = tf.math.maximum(best_ious, ious)
            iou_mask = tf.cast(best_ious < 0.7, tf.float32)

            obj_loss += tf.math.reduce_sum(tf.math.square(1 - objectness) * truth_mask[..., ir])
            obj_loss += tf.math.reduce_sum(tf.math.square(objectness) * inv_truth_mask[..., ir] * iou_mask)

        tot_box_loss += box_loss
        tot_obj_loss += obj_loss
        tot_cls_loss += cls_loss
        #print(box_loss, obj_loss, cls_loss)

    return tot_box_loss, tot_obj_loss, tot_cls_loss



img, input = read_img("test_img/doggos.jpg", 608)
gt_boxes = [[0.7155, 0.539, 0.385, 0.572],
            [0.363, 0.557, 0.452, 0.606]]
gt_count = 2

anchor_sizes = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
scales = [1.2, 1.1, 1.05]

print(input.shape)
# input = np.random.random([1, 608, 608, 3])
model = YOLOv4Model()
model.load_weights("yolov4.weights")

output = model(input)
print(calc_loss(gt_boxes, [16, 16], output))
