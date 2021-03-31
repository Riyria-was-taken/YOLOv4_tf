from model import YOLOv4Model
import numpy as np

from img import read_img, draw_img, save_img, add_bboxes
import inference
import train

import math
import sys


def run_infer(weights_file, labels_file, image_path, out_filename):

    model = YOLOv4Model()
    model.load_weights(weights_file)

    img, input = read_img(image_path, 608)

    cls_names = open(labels_file, "r").read().split("\n")

    boxes, scores, labels = inference.infer(model, cls_names, input)

    pixels = add_bboxes(img, boxes, scores, labels)
    if out_filename:
        save_img(out_filename, pixels)
    else:
        draw_img(pixels)

def run_training(file_root, annotations, batch_size, steps):

    model = train.train(file_root, annotations, batch_size, steps)

    img, input = read_img("test_img/doggos.jpg", 608)
    cls_names = open("coco-labels.txt", "r").read().split("\n")
    boxes, scores, labels = inference.infer(model, cls_names, input)
    pixels = add_bboxes(img, boxes, scores, labels)
    draw_img(pixels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True
    parser_infer = subparsers.add_parser("infer")
    parser_infer.add_argument("--weights", "-w", nargs="?", default="yolov4.weights")
    parser_infer.add_argument("--classes", "-c", nargs="?", default="coco-labels.txt")
    parser_infer.add_argument("--output", "-o")
    parser_infer.add_argument("image")  # , nargs="+")
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("file_root")
    parser_train.add_argument("annotations")
    parser_train.add_argument("--batch_size", "-b", default="32")
    parser_train.add_argument("--steps", "-s", default="100")
    subparsers.add_parser("verify")

    args = parser.parse_args()

    if args.action == "infer":
        run_infer(args.weights, args.classes, args.image, args.output)
    elif args.action == "train":
        batch_size = int(args.batch_size)
        steps = int(args.steps)
        run_training(args.file_root, args.annotations, batch_size, steps)
    else:
        print("The " + args.action + " action is not yet implemented :<")
