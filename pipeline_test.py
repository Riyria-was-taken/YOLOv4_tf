import os

from pipeline import YOLOv4Pipeline
from img import draw_img, draw_bboxes

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
    output_input=True
)

pipeline.build()
images, bboxes, batch_img, batch_boxes = pipeline.run()
print(bboxes.at(0))

# for i, image in enumerate(images):
#    plt.imshow(image)
#    plt.savefig('image' + str(i) + '.png')
#    plt.clf()
labels = ["lol" for i in range(len(bboxes.at(0)))]
scores = [1.0 for i in range(len(bboxes.at(0)))]



#for i in range(len(batch_img)):
#    draw_img(draw_bboxes(batch_img.at(i), batch_boxes.at(i), image_size))


for i in range(len(images)):
    draw_img(draw_bboxes(images.at(i), bboxes.at(i), image_size))