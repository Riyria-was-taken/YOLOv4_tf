from model import YOLOv4Model
import numpy as np

input = np.random.random([1, 608, 608, 3])
model = YOLOv4Model((608, 608))
for x in model(input):
    print(x.shape)
