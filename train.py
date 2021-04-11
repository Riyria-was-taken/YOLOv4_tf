from model import YOLOv4Model, calc_loss
import numpy as np
import tensorflow as tf

from img import read_img, draw_img
from pipeline import YOLOv4Pipeline
import utils

import math
import os


SET_MEMORY_GROWTH = True






def train(file_root, annotations_file, batch_size, epochs, steps_per_epoch, use_gpu):

    if SET_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = YOLOv4Model()
    #model.load_weights("yolov4.weights")

    image_size = (608, 608)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), "little")

    pipeline = YOLOv4Pipeline(
        file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed, use_gpu
    )
    dataset = pipeline.dataset()

    model.model.compile(
        optimizer=tf.keras.optimizers.Adam()
    )
    model.model.fit(pipeline.dataset(), epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model





'''
model.model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tuple([lambda x, y : calc_loss(i, x, y) for i in range(3)]),
    loss_weights=[1.0, 1.0, 1.0]
)
model.model.fit(pipeline.dataset(), epochs=5, steps_per_epoch=10)
'''
