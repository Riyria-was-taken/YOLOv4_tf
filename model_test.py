from model import YOLOv4Model
import numpy as np

input = np.random.random([1, 608, 608, 3])
model = YOLOv4Model()
print(model.summary())
model.load_weights('yolov4.weights')
for x in model(input):
    print(x)
