import tensorflow as tf
import cv2

path = 'birb.jpg'
size = 416

img = tf.image.decode_image(open(path, 'rb').read(), channels=3)
pix = img.numpy()

img = tf.image.resize(img, (size, size)) / 255
# do detection

pix = cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
cv2.imshow('A birb', pix)
cv2.waitKey(0)
