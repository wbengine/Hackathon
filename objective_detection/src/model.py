import numpy as np
import json

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 模型路径
PATH_TO_CKPT = '../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

# label字典路径，用于识别出物品后展示类别名
PATH_TO_LABELS = '../../tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90  # 最大分类数量


class Model(object):
    def __init__(self):
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)  # 获得类别字典
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=NUM_CLASSES,
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.sess = None
        self.inputs = {}
        self.outputs = {}

    def prepare(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():  # 加载模型
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            self.inputs['image'] = image_tensor
            self.outputs['boxes'] = boxes
            self.outputs['scores'] = scores
            self.outputs['classes'] = classes
            self.outputs['num_detections'] = num_detections

        print('run demo to init the graph')
        self.run(np.zeros(shape=[800, 800, 3], dtype=np.uint8))

        return self

    def run(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        outputs = self.sess.run(
            self.outputs,
            feed_dict={self.inputs['image']: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(outputs['boxes']),
            np.squeeze(outputs['classes']).astype(np.int32),
            np.squeeze(outputs['scores']),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3)

        return self.process_outputs(outputs), image

    def process_outputs(self, outputs):
        boxes = np.squeeze(outputs['boxes'])
        scores = np.squeeze(outputs['scores'])
        classes = np.squeeze(outputs['classes']).astype(np.int32)

        res_list = []
        for box, class_id, s in zip(boxes, classes, scores):
            if s <= 0.5:
                continue
            class_info = self.get_class_info(class_id)
            res_list.append(
                {
                    'y1': float(box[0]),
                    'x1': float(box[1]),
                    'y2': float(box[2]),
                    'x2': float(box[3]),
                    'text': class_info['name'],
                    'rate': float(s)
                }
            )
        return res_list

    def get_class_info(self, class_id):
        return self.category_index[class_id]


if __name__ == '__main__':
    import cv2

    m = Model()
    m.prepare()

    data = open('../images/panda.jpg', 'rb').read()
    array = np.frombuffer(data, dtype='uint8')
    frame = cv2.imdecode(array, 1)
    # cv2.imshow("window", frame)

    res_list, frame = m.run(frame)

    print(json.dumps(res_list, indent=2))

    cv2.imshow('Video', frame)  # 展示已标记物体的图片
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()
