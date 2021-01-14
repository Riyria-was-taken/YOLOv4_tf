import tensorflow as tf
import cv2


def read_img(path, size):
    img = tf.image.decode_image(open(path, "rb").read(), channels=3)
    pixels = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    img = tf.image.resize(img, (size, size)) / 255
    img = tf.reshape(img, (1, size, size, 3))
    return (pixels, img)


def draw_img(pixels, boxes, scores, classes):
    (h, w, _) = pixels.shape
    for i in range(len(boxes)):
        (x1, y1, x2, y2) = boxes[i]
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        pixels = cv2.rectangle(pixels, p1, p2, (255, 0, 0), 2)
        label = classes[i] + ": " + str(scores[i])
        t_size = cv2.getTextSize(label, 0, 0.5, thickness=int(0.6 * (h + w) / 600) // 2)[0]
        cv2.rectangle(pixels, p1, (p1[0] + t_size[0], p1[1] - t_size[1] - 3), (255, 0, 0), -1)
        cv2.putText(
            pixels,
            label,
            (p1[0], p1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            int(0.6 * (h + w) / 600) // 2,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow("Image", pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
(pix, _) = read_img('birb.jpg', 300)
boxes = [[0.14, 0.27, 0.61, 1.00]]
scores = [0.69]
classes = [14.0]
draw_img(pix, boxes, scores, classes)
"""
