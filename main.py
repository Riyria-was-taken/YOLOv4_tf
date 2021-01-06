from model import YOLOv4Model
import numpy as np

from img import read_img, draw_img

import math
import sys


def sigmoid(x):
    return 1 / (1 + math.exp(-x))



#input = np.random.random([1, 608, 608, 3])




anchor_sizes = [
    [(12,16), (19,36), (40,28)],
    [(36,75), (76,55), (72,146)],
    [(142,110), (192,243), (459,401)]
]
scales = [1.2, 1.1, 1.05]

def run_infer(image_path, weights_file, labels_file):

    model = YOLOv4Model()
    model.load_weights(weights_file)

    img, input = read_img(image_path, 608)

    cls_names = open(labels_file, 'r').read().split('\n')

    pred_boxes = [[] for i in range(len(cls_names))]
    for i, preds in enumerate(model(input)):
        s = preds.shape
        gw, gh = s[1 : 3]
        d = s[3]
        for ix in range(gw):
            for iy in range(gh):
                for ir in range(3):

                    data = preds[0, iy, ix, (d // 3) * ir : (d // 3) * (ir + 1)]

                    dx, dy, dw, dh = data[ : 4]
                    objectness = data[4]
                    confidence = data[5 : ]

                    if (objectness > 0):
                        cls = np.argmax(confidence)

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

    #nms
    def iou(box1, box2):
        l = max(box1[0], box2[0])
        t = max(box1[1], box2[1])
        r = min(box1[2], box2[2])
        b = min(box1[3], box2[3])
        i = max(0, (r - l) * (b - t))
        u = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
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
    draw_img(img, boxes, scores, labels)




if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == 'infer':
        run_infer(*sys.argv[2 : 5])
    else:
        print("Usage:")
        print(sys.argv[0], 'infer', '[image path]', '[yolo weights file]', '[class names file]')