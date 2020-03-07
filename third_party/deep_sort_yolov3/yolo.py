#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

# 当前文件夹的母文件夹加入系统路径
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np

# from keras import backend as K                            # tf新版本由下面两行代替
# from keras.models import load_model                       # tf新版本由下面两行代替
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

# # 设置使用的gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)


# 使用绝对路径，避免不同路径下引用时出错====================================================================================
deep_sort_yolov3_root_dir = os.path.dirname(os.path.abspath(__file__))
model_data_path = os.path.join(deep_sort_yolov3_root_dir, 'model_data')
ModelPath = os.path.join(model_data_path, 'yolo_608.h5')
AnchorsPath = os.path.join(model_data_path, 'yolo_anchors.txt')
ClassesPath = os.path.join(model_data_path, 'coco_classes.txt')
# 结束==================================================================================================================

class YOLO(object):
    def __init__(self):
        # 更改绝对路径===================================================================================================
        self.model_path = ModelPath
        self.anchors_path = AnchorsPath
        self.classes_path = ClassesPath
        # 结束==========================================================================================================
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (608, 608)  # fixed size or (None, None) or （416, 416）
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person':
                continue
            box = out_boxes[i]
            # score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxes.append([x, y, w, h])

        return return_boxes

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    # import sys
    # from os import path
    # sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    yolo = YOLO()