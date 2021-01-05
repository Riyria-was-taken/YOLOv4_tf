from model import YOLOv4Model
import numpy as np

from img import read_img, draw_img

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



img, input = read_img('test_img/doggos.jpg', 608)

print(input.shape)
#input = np.random.random([1, 608, 608, 3])
model = YOLOv4Model()
print(model.summary())
model.load_weights('yolov4.weights')

anchor_sizes = [
    [(12,16), (19,36), (40,28)],
    [(36,75), (76,55), (72,146)],
    [(142,110), (192,243), (459,401)]
]
scales = [1.2, 1.1, 1.05]
pred_boxes = []
for iscale, preds in enumerate(model(input)):
    s = preds.shape
    gw, gh = s[1 : 3]
    d = s[3]
    for i in range(gw):
        for j in range(gh):
            for ij in range(3):
                dx, dy, dw, dh = preds[0, i, j, (d // 3) * ij : (d // 3) * ij + 4]
                confidence = preds[0, i, j, (d // 3) * ij + 4]
                if (confidence > 0):
                    stride_x = 1 / gw
                    stride_y = 1 / gh
                    x = (sigmoid(dx) * scales[iscale] - 0.5 * (scales[iscale] - 1) + j) * stride_x
                    y = (sigmoid(dy) * scales[iscale] - 0.5 * (scales[iscale] - 1) + i) * stride_y

                    w, h = anchor_sizes[iscale][ij]
                    w /= 608
                    h /= 608
                    w *= math.exp(dw)
                    h *= math.exp(dh)

                    l, t, r, b = x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h
                    pred_boxes.append([l, t, r, b])
                    print(l, t, r, b)



scores = [0.69]
classes = [14.0]
draw_img(img, pred_boxes, scores, classes)
