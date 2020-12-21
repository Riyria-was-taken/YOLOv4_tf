from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf


class YOLOv4Pipeline(Pipeline):
    def __init__(self, file_root, annotations_file,
                 batch_size, image_size,
                 num_threads, device_id, seed):
        super(YOLOv4Pipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed
        )

        self.image_size = image_size

        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            ltrb = True,
            shard_id = device_id,
            num_shards = num_threads,
            ratio = True,
            random_shuffle = True
        )
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

        self.crop = ops.RandomBBoxCrop(
            bbox_layout="xyXY",
            allow_no_crop=False
        )

        self.resize = ops.Resize(resize_x=image_size[0], resize_y=image_size[1])
        self.gen_perm = ops.BatchPermutation()
        self.shuffle = ops.PermuteBatch()
        self.mosaic_uniform = ops.Uniform(range=(0.2, 0.8))
        self.zeros = ops.Constant(fdata=0.0, shape=[])
        self.slice = ops.Slice()
        self.cat = ops.Cat()
        self.stack = ops.Stack()
        self.cast = ops.Cast(dtype=types.DALIDataType.INT32)

        self.coord_transform = ops.CoordTransform()
        self.transform_crop = ops.transforms.Crop()

    def define_graph(self):
        inputs, bboxes, labels = self.input() # skip_crowd_during_training
        images = self.decode(inputs)
        images = self.resize(images)

        images, bboxes, labels = self.mosaic(images, bboxes, labels)

        return images, bboxes, labels

    def permute(self, images, bboxes, labels):
        indices = self.gen_perm()
        return (self.shuffle(images, indices=indices),
            self.shuffle(bboxes, indices=indices),
            self.shuffle(labels, indices=indices))

    def generate_tiles(self, images, bboxes, labels, shape_x, shape_y):
        images, bboxes, labels = self.permute(images, bboxes, labels)
        crop_anchor, crop_shape, bboxes, labels = self.crop(bboxes, labels,
            crop_shape=self.stack(shape_x, shape_y),
            input_shape=self.image_size)
        crop_anchor = self.cast(crop_anchor)
        crop_shape = self.cast(crop_shape)
        images = self.slice(images, crop_anchor, crop_shape)
        return images, bboxes, labels

    def bbox_adjust(self, bboxes, shape_x, shape_y, pos_x, pos_y):
        sx, sy, ex, ey = pos_x, pos_y, shape_x + pos_x, shape_y + pos_y
        MT = self.transform_crop(
            to_start=self.stack(sx, sy, sx, sy),
            to_end=self.stack(ex, ey, ex, ey)
        )
        return self.coord_transform(bboxes, MT=MT)

    def mosaic(self, images, bboxes, labels):
        prob_x = self.mosaic_uniform()
        prob_y = self.mosaic_uniform()

        pix0_x = self.cast(prob_x * self.image_size[0])
        pix0_y = self.cast(prob_y * self.image_size[0])
        pix1_x = self.image_size[0] - pix0_x
        pix1_y = self.image_size[1] - pix0_y

        images00, bboxes00, labels00 = self.generate_tiles(images, bboxes, labels,
            pix0_x, pix0_y)
        images01, bboxes01, labels01 = self.generate_tiles(images, bboxes, labels,
            pix0_x, pix1_y)
        images10, bboxes10, labels10 = self.generate_tiles(images, bboxes, labels,
            pix1_x, pix0_y)
        images11, bboxes11, labels11 = self.generate_tiles(images, bboxes, labels,
            pix1_x, pix1_y)
        images0 = self.cat(images00, images01, axis=0)
        images1 = self.cat(images10, images11, axis=0)
        images = self.cat(images0, images1, axis=1)

        bboxes00 = self.bbox_adjust(bboxes00, prob_x, prob_y, self.zeros(), self.zeros())
        bboxes01 = self.bbox_adjust(bboxes01, prob_x, 1.0 - prob_y, self.zeros(), prob_y)
        bboxes10 = self.bbox_adjust(bboxes10, 1.0 - prob_x, prob_y, prob_x, self.zeros())
        bboxes11 = self.bbox_adjust(bboxes11, 1.0 - prob_x, 1.0 - prob_y, prob_x, prob_y)
        bboxes = self.cat(bboxes00, bboxes01, bboxes10, bboxes11)

        labels = self.cat(labels00, labels01, labels10, labels11)

        return images, bboxes, labels
