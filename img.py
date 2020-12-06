import tensorflow as tf
import cv2

def read_img(path, size):
	img = tf.image.decode_image(open(path, 'rb').read(), channels=3)
	pixels = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
	img = tf.image.resize(img, (size, size)) / 255
	data = tf.data.Dataset.from_tensors(img)
	return (pixels, data)

def draw_img(pixels, boxes, scores, classes):
	cv2.imshow('Image', pixels)
	cv2.waitKey(0)
