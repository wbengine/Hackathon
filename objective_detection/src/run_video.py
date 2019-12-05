#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Queue, Pool
from absl import flags, logging, app
from video_stream import WebcamVideoStream

# tensorflow api 接口相关函数
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# flags.DEFINE_string('model_path', 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb',
#                     'the frozen model. *.pb file')
# flags.DEFINE_string('label_path', 'tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt',
#                     'the label path to show the objective name')
# flags.DEFINE_integer('num_classes', 90,
#                      'the number of classes')

flags.DEFINE_integer('num_workers', 2, 'worker number to compute Objective Detection')
flags.DEFINE_integer('max_queue_size', 10, 'max queue size')
flags.DEFINE_integer('image_width', 600, 'image width')
flags.DEFINE_integer('image_height', None, 'image height')
flags.DEFINE_integer('video_source', 0, 'the video source id')
FLAGS = flags.FLAGS

# 模型路径
PATH_TO_CKPT = '../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

# label字典路径，用于识别出物品后展示类别名
PATH_TO_LABELS = '../tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'
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
    return image_np


def worker(input_q, output_q):
    detection_graph = tf.Graph()
    with detection_graph.as_default():  # 加载模型
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    while True:  # 全局变量input_q与output_q定义，请看下文
        frame = input_q.get()  # 从多进程输入队列，取值
        if frame is None:
            break
        output_q.put(detect_objects(frame, sess, detection_graph))  # detect_objects函数 返回一张图片，标记所有被发现的物品
    sess.close()


def main(_):
    input_q = Queue(maxsize=FLAGS.max_queue_size)  # 多进程输入队列
    output_q = Queue(maxsize=FLAGS.max_queue_size)  # 多进程输出队列
    pool = Pool(FLAGS.num_workers, worker, (input_q, output_q))  # 多进程加载模型

    video_capture = WebcamVideoStream(src=FLAGS.video_source,
                                      width=FLAGS.image_width,
                                      height=FLAGS.image_height).start()

    while True:
        frame = video_capture.read()  # video_capture多线程读取视频流
        input_q.put(frame)  # 视频帧放入多进程输入队列
        frame = output_q.get()  # 多进程输出队列取出标记好物体的图片

        cv2.imshow('Video', frame)  # 展示已标记物体的图片
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()  # 关闭多进程
    video_capture.stop()  # 关闭视频流
    cv2.destroyAllWindows()  # opencv窗口关闭


if __name__ == '__main__':
    app.run(main)
