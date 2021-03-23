from model import YOLOv4Model
import numpy as np

from img import read_img, draw_img, save_img, add_bboxes
import inference

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True
    infer = subparsers.add_parser("infer")
    infer.add_argument("--weights", "-w", nargs="?", default="yolov4.weights")
    infer.add_argument("--classes", "-c", nargs="?", default="coco-labels.txt")
    infer.add_argument("--output", "-o")
    infer.add_argument("image")  # , nargs="+")
    subparsers.add_parser("train")
    subparsers.add_parser("verify")

    args = parser.parse_args()

    if args.action == "infer":
        run_infer(args.weights, args.classes, args.image, args.output)
    else:
        print("The " + args.action + " action is not yet implemented :<")
