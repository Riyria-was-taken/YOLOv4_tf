import os

import tensorflow as tf

from pipeline import YOLOv4Pipeline
from img import add_bboxes, draw_img

dali_extra = os.environ["DALI_EXTRA_PATH"]
file_root = os.path.join(dali_extra, "db", "coco", "images")
annotations_file = os.path.join(dali_extra, "db", "coco", "instances.json")

batch_size = 8
image_size = (608, 608)
num_threads = 1
device_id = 0
seed = int.from_bytes(os.urandom(4), "little")

pipeline = YOLOv4Pipeline(
    file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed,
)

dataset = pipeline.dataset()
print(tf.data.experimental.get_single_element(dataset.take(1)))
