import nvidia.dali as dali
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
import ops


class YOLOv4Pipeline:
    def __init__(
        self, file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed
    ):

        self._batch_size = batch_size
        self._image_size = image_size
        self._file_root = file_root
        self._annotations_file = annotations_file

        self._num_threads = num_threads
        self._device_id = device_id

        self._pipe = dali.pipeline.Pipeline(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
        )
        self._define_pipeline()

    def _define_pipeline(self):
        with self._pipe:
            images, bboxes, labels = ops.input(
                self._file_root, self._annotations_file, self._device_id, self._num_threads
            )
            images = dali.fn.resize(
                images, resize_x=self._image_size[0], resize_y=self._image_size[1]
            )

            images, bboxes, labels = ops.mosaic(images, bboxes, labels, self._image_size)

            images = dali.fn.cast(images, dtype=dali.types.FLOAT) / 255.0
            labels = dali.fn.cast(
                dali.fn.transpose(dali.fn.stack(labels), perm=[1, 0]),
                dtype=dali.types.FLOAT
            )

            self._pipe.set_outputs(images, dali.fn.cat(bboxes, labels, axis=1))

    def dataset(self):
        output_shapes = ((self._batch_size, self._image_size[0], self._image_size[0], 3), (self._batch_size, None, 5))
        output_dtypes = (tf.float32, tf.float32)
        return dali_tf.DALIDataset(
            pipeline=self._pipe,
            batch_size=self._batch_size,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            device_id=0
        )

    def build(self):
        self._pipe.build()

    def run(self):
        return self._pipe.run()
