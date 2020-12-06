import tensorflow as tf
import cv2

def read_img(path, size):
    img = tf.image.decode_image(open(path, 'rb').read(), channels=3)
    pixels = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    img = tf.image.resize(img, (size, size)) / 255
    data = tf.data.Dataset.from_tensors(img)
    return (pixels, data)

def draw_img(pixels, boxes, scores, classes):
    (h, w, _) = pixels.shape
    for i in range(len(boxes)):
        (x1, y1, x2, y2) = boxes[i]
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        pixels = cv2.rectangle(pixels, p1, p2, (255, 0, 0), 2)
    cv2.imshow('Image', pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
(pix, _) = read_img('birb.jpg', 300)
boxes = [[0.14, 0.27, 0.61, 1.00]]
scores = [0.69]
classes = [14.0]
draw_img(pix, boxes, scores, classes)
'''
