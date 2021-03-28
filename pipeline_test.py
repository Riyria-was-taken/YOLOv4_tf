import os

from pipeline import YOLOv4Pipeline
from img import add_bboxes, draw_img

dali_extra = os.environ["DALI_EXTRA_PATH"]
file_root = os.path.join(dali_extra, "db", "coco", "images")
annotations_file = os.path.join(dali_extra, "db", "coco", "instances.json")

batch_size = 8
image_size = (256, 256)
num_threads = 1
device_id = 0
seed = int.from_bytes(os.urandom(4), "little")

pipeline = YOLOv4Pipeline(
    file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed,
)

pipeline.build()
images, bboxes = pipeline.run()


for i in range(len(images)):
    labels = ["lol" for i in range(len(bboxes.at(i)))]
    scores = [1.0 for i in range(len(bboxes.at(i)))]
    draw_img(add_bboxes(images.at(i), bboxes.at(i)[:, :4], scores, labels))
