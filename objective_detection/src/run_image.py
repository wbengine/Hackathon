#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Queue, Pool
from io import BytesIO

# tensorflow api 接口相关函数
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 模型路径
PATH_TO_CKPT = '../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

# label字典路径，用于识别出物品后展示类别名
PATH_TO_LABELS = '../../tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90  # 最大分类数量
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  # 获得类别字典
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 物体识别神经网络，向前传播获得识别结果
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3)

    print('boxes=', np.squeeze(boxes))
    print('classes=', np.squeeze(classes).astype(np.int32))
    print('scores=', np.squeeze(scores))

    return image_np


def main(_):
    detection_graph = tf.Graph()
    with detection_graph.as_default():  # 加载模型
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

        # read image
        data = open('../images/panda.jpg', 'rb').read()
        array = np.frombuffer(data, dtype='uint8')
        frame = cv2.imdecode(array, 1)
        # cv2.imshow("window", frame)

        frame = detect_objects(frame, sess, detection_graph)

        cv2.imshow('Video', frame)  # 展示已标记物体的图片
        cv2.waitKey(0)
        # Destroying present windows on screen
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(0)
