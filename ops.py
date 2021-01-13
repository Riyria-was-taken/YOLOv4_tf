import nvidia.dali as dali

def input(file_root, annotations_file, device_id, num_threads, random_shuffle=True):
    inputs, bboxes, classes = dali.fn.coco_reader(
        file_root = file_root,
        annotations_file = annotations_file,
        ltrb = True,
        shard_id = device_id,
        num_shards = num_threads,
        ratio = True,
        random_shuffle = random_shuffle
    )
    images = dali.fn.image_decoder(inputs, device = 'cpu', output_type = dali.types.RGB)
    return images, bboxes, classes

def permute(images, bboxes, labels):
    indices = dali.fn.batch_permutation()
    return (dali.fn.permute_batch(images, indices=indices),
            dali.fn.permute_batch(bboxes, indices=indices),
            dali.fn.permute_batch(labels, indices=indices))

def generate_tiles(images, bboxes, labels, shape_x, shape_y, image_size):
    images, bboxes, labels = permute(images, bboxes, labels)
    crop_anchor, crop_shape, bboxes, labels = dali.fn.random_bbox_crop(
        bboxes, labels,
        crop_shape = dali.fn.stack(shape_x, shape_y),
        input_shape = image_size,
        bbox_layout = "xyXY",
        allow_no_crop = False
    )
    images = dali.fn.slice(images, crop_anchor, crop_shape)
    return images, bboxes, labels

def bbox_adjust(bboxes, shape_x, shape_y, pos_x, pos_y):
    sx, sy, ex, ey = pos_x, pos_y, shape_x + pos_x, shape_y + pos_y
    MT = dali.fn.transforms.crop(
        to_start = dali.fn.stack(sx, sy, sx, sy),
        to_end = dali.fn.stack(ex, ey, ex, ey)
    )
    return dali.fn.coord_transform(bboxes, MT = MT)

def mosaic(images, bboxes, labels, image_size):
    prob_x = dali.fn.uniform(range = (0.2, 0.8))
    prob_y = dali.fn.uniform(range = (0.2, 0.8))

    pix0_x = dali.fn.cast(prob_x * image_size[0], dtype = dali.types.INT32)
    pix0_y = dali.fn.cast(prob_y * image_size[1], dtype = dali.types.INT32)
    pix1_x = image_size[0] - pix0_x
    pix1_y = image_size[1] - pix0_y

    images00, bboxes00, labels00 = generate_tiles(images, bboxes, labels,
        pix0_x, pix0_y, image_size)
    images01, bboxes01, labels01 = generate_tiles(images, bboxes, labels,
        pix0_x, pix1_y, image_size)
    images10, bboxes10, labels10 = generate_tiles(images, bboxes, labels,
        pix1_x, pix0_y, image_size)
    images11, bboxes11, labels11 = generate_tiles(images, bboxes, labels,
        pix1_x, pix1_y, image_size)
    images0 = dali.fn.cat(images00, images01, axis = 0)
    images1 = dali.fn.cat(images10, images11, axis = 0)
    images = dali.fn.cat(images0, images1, axis = 1)

    zeros = dali.types.Constant(0.0)
    bboxes00 = bbox_adjust(bboxes00, prob_x, prob_y, zeros, zeros)
    bboxes01 = bbox_adjust(bboxes01, prob_x, 1.0 - prob_y, zeros, prob_y)
    bboxes10 = bbox_adjust(bboxes10, 1.0 - prob_x, prob_y, prob_x, zeros)
    bboxes11 = bbox_adjust(bboxes11, 1.0 - prob_x, 1.0 - prob_y, prob_x, prob_y)
    bboxes = dali.fn.cat(bboxes00, bboxes01, bboxes10, bboxes11)

    labels = dali.fn.cat(labels00, labels01, labels10, labels11)

    return images, bboxes, labels
