from keras.models import Sequential
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import cv2


class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w * self.h + box.w * box.h - intersection
        return intersection / union

    def intersect(self, box):
        width = self.__overlap([self.x - self.w / 2, self.x + self.w / 2], [box.x - box.w / 2, box.x + box.w / 2])
        height = self.__overlap([self.y - self.h / 2, self.y + self.h / 2], [box.y - box.h / 2, box.y + box.h / 2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def load_weights(model, weights_file):

    weight_reader = WeightReader(weights_file)

    weight_reader.reset()

    for i in range(len(model.layers)):
        if 'conv' in model.layers[i].name:
            if 'batch' in model.layers[i + 1].name:
                norm_layer = model.layers[i + 1]
                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean = weight_reader.read_bytes(size)
                var = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            conv_layer = model.layers[i]
            if len(conv_layer.get_weights()) > 1:
                bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])


def prediction_bboxes(netout, class_names, anchors, prob_threshold, nms_threshold):
    """
    takes network output and returns set of bounding boxes
    :param netout:
    :param anchors:
    :param class_names:
    :param prob_threshold:
    :param nms_threshold:
    :return bboxes: array of class, box (xmin, xmax, ymin, ymax), and probability
    """

    # fixed for now - eventually read from .cfg file
    NORM_H, NORM_W = 416, 416
    GRID_H, GRID_W = 13, 13
    BOX = 5
    CLASS = 20
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]

    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[0, row, col, b, :5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[0, row, col, b, 5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > prob_threshold

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].probs[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= nms_threshold:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    bboxes = []
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]

        if max_prob > prob_threshold:
            bboxes.append([class_names[max_indx], (box.x*NORM_W, box.y*NORM_H, box.w*NORM_W, box.h*NORM_H), max_prob])

    return bboxes


def sigmoid(x):
    return 1. / (1.  + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def tiny_yolo_voc():
    """
    create and return Tiny YOLO v2 model
    :return model:
    """

    # fixed for now - eventually read from .cfg file
    GRID_H, GRID_W = 13, 13
    BOX = 5
    CLASS = 20

    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=(416, 416, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2 - 5
    for i in range(0, 4):
        model.add(Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Layer 7 - 8
    for _ in range(0, 2):
        model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    # Layer 9
    model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))

    return model


if __name__ == "__main__":
    model = tiny_yolo_voc()
    load_weights(model, "darknet/tiny-yolo-voc.weights")
    model.summary()

    rimg = cv2.imread("test_images/test1.jpg")

    img = cv2.resize(rimg, (416, 416))
    img = img / 255.
    img = img[:, :, ::-1]
    img = np.expand_dims(img, 0)

    netout = model.predict(img)

    print(netout.shape)

    bboxes = prediction_bboxes(netout, rimg.shape, 0.2, 0.4)

    print(bboxes)