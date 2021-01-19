from model import YOLOv4Model
import numpy as np

from img import read_img, draw_img, save_img, add_bboxes

import math
import sys


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


anchor_sizes = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
scales = [1.2, 1.1, 1.05]


def run_infer(weights_file, labels_file, image_path, out_filename):

    model = YOLOv4Model()
    model.load_weights(weights_file)

    img, input = read_img(image_path, 608)

    cls_names = open(labels_file, "r").read().split("\n")

    pred_boxes = [[] for i in range(len(cls_names))]
    for i, preds in enumerate(model(input)):
        s = preds.shape
        print(s)
        gw, gh = s[1:3]
        d = s[3]
        for ix in range(gw):
            for iy in range(gh):
                for ir in range(3):
                    data = preds[0, iy, ix, (d // 3) * ir : (d // 3) * (ir + 1)]

                    dx, dy, dw, dh = data[:4]
                    confidence = [sigmoid(x) for x in data[5:]]
                    cls = np.argmax(confidence)
                    objectness = confidence[cls] * sigmoid(data[4])

                    if objectness > 0.25:
                        stride_x = 1 / gw
                        stride_y = 1 / gh
                        x = (sigmoid(dx) * scales[i] - 0.5 * (scales[i] - 1) + ix) * stride_x
                        y = (sigmoid(dy) * scales[i] - 0.5 * (scales[i] - 1) + iy) * stride_y

                        w, h = anchor_sizes[i][ir]
                        w /= 608
                        h /= 608
                        w *= math.exp(dw)
                        h *= math.exp(dh)

                        l, t, r, b = x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h
                        pred_boxes[cls].append((confidence[cls] * objectness, [l, t, r, b]))

    # nms
    def iou(box1, box2):
        l = max(box1[0], box2[0])
        t = max(box1[1], box2[1])
        r = min(box1[2], box2[2])
        b = min(box1[3], box2[3])
        i = max(0, (r - l) * (b - t))
        u = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - i
        return i / u

    boxes = []
    scores = []
    labels = []
    for cls in range(len(cls_names)):
        cls_preds = sorted(pred_boxes[cls])
        while len(cls_preds) > 0:
            score, box = cls_preds[-1]
            boxes.append(box)
            scores.append(score)
            labels.append(cls_names[cls])
            rem = []
            for score2, box2 in cls_preds:
                if iou(box, box2) < 0.213:
                    rem.append((score2, box2))
            cls_preds = rem

    print(scores)
    pixels = add_bboxes(img, boxes, scores, labels)
    if out_filename:
        save_img(out_filename, pixels)
    else:
        draw_img(pixels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True
    infer = subparsers.add_parser("infer")
    infer.add_argument("--weights", "-w", nargs="?", default="yolov4.weights")
    infer.add_argument("--classes", "-c", nargs="?", default="coco-labels.txt")
    infer.add_argument("--output", "-o")
    infer.add_argument("image")  # , nargs="+")
    subparsers.add_parser("train")
    subparsers.add_parser("verify")

    args = parser.parse_args()

    if args.action == "infer":
        run_infer(args.weights, args.classes, args.image, args.output)
    else:
        print("The " + args.action + " action is not yet implemented :<")
