import nvidia.dali as dali
import ops

class YOLOv4Pipeline():
    def __init__(self, file_root, annotations_file,
                 batch_size, image_size,
                 num_threads, device_id, seed):

        self._batch_size = batch_size
        self._image_size = image_size
        self._file_root = file_root
        self._annotations_file = annotations_file

        self._num_threads = num_threads
        self._device_id = device_id

        self._pipe = dali.pipeline.Pipeline(
            batch_size = batch_size,
            num_threads = num_threads,
            device_id = device_id,
            seed = seed
        )
        self._define_pipeline()

    def _define_pipeline(self):
        with self._pipe:
            images, bboxes, labels = ops.input(
                self._file_root,
                self._annotations_file,
                self._device_id,
                self._num_threads
            )
            images = dali.fn.resize(
                images,
                resize_x = self._image_size[0],
                resize_y = self._image_size[1]
            )

            images, bboxes, labels = ops.mosaic(images, bboxes, labels, self._image_size)

            self._pipe.set_outputs(images, bboxes, labels)

    def build(self):
        self._pipe.build()

    def run(self):
        return self._pipe.run()
