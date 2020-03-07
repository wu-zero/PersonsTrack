__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

# 当前文件夹的母文件夹加入系统路径
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))


import os
import warnings
import cv2
import numpy as np
from PIL import Image

from .deep_sort import preprocessing
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .tools import generate_detections as gdet
warnings.filterwarnings('ignore')

# 使用绝对路径，避免不同路径下引用时出错====================================================================================
deep_sort_yolov3_root_dir = os.path.dirname(os.path.abspath(__file__))
model_data_path = os.path.join(deep_sort_yolov3_root_dir, 'model_data')
BoxEncoderModelAddress = os.path.join(model_data_path, 'mars-small128.pb')


class DeepSortPreprocess:
    def __init__(self, model_address=BoxEncoderModelAddress):
        self.encoder = gdet.create_box_encoder(model_address, batch_size=1)

    def get_features(self, frame, boxs):
        features = self.encoder(frame, boxs)
        return features


class DeepSort:
    def __init__(self):
        # 非极大值抑制参数
        self.nms_max_overlap = 1.0

        # deep_sort参数
        max_cosine_distance = 0.3
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=0.7, max_age=15, n_init=3)

    def update(self, box_list, feature_list, other_data_list=None):
        if other_data_list is None:
            other_data_list = [None] * len(box_list)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature, other_data) for bbox, feature, other_data in zip(box_list, feature_list, other_data_list)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        track_new_id_list, track_delete_id_list = self.tracker.update(detections)

        not_confirmed_detected_track_list = []
        detected_track_list = []

        for track in self.tracker.tracks:
            if not track.is_confirmed():
                if track.time_since_update == 0:
                    not_confirmed_detected_track_list.append({'trackID': track.track_id,
                                                              'body_box': track.bbox,
                                                              'other_data': track.other_data})
            else:
                if track.time_since_update == 0:
                    detected_track_list.append({'trackID': track.track_id,
                                                'body_box': track.bbox,
                                                'other_data': track.other_data})

        return track_new_id_list, track_delete_id_list, not_confirmed_detected_track_list, detected_track_list


if __name__ == '__main__':
    from yolo import YOLO
    yolo = YOLO()
    deepsort_preprocess = DeepSortPreprocess()
    deepsort = DeepSort()

    video_capture = cv2.VideoCapture('../data/PETS09-S2L1.mp4')
    frame_i = -1
    while True:
        frame_i += 1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret is not True:
            break

        # 检测框
        image_for_yolo = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image_for_yolo)

        features = deepsort_preprocess.get_features(frame, boxs)
        # 追踪
        track_new_id_list, track_delete_id_list, not_confirmed_detected_track, detected_track = deepsort.update(boxs, features)

        for track_data in not_confirmed_detected_track:
            track_id = track_data['trackID']
            track_bbox = track_data['body_box']
            cv2.rectangle(frame, (int(track_bbox[0]), int(track_bbox[1])), (int(track_bbox[2]), int(track_bbox[3])), (255, 0, 255), 2)
            cv2.putText(frame, str(track_id), (int(track_bbox[0]), int(track_bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for track_data in detected_track:
            track_id = track_data['trackID']
            track_bbox = track_data['body_box']
            cv2.rectangle(frame, (int(track_bbox[0]), int(track_bbox[1])), (int(track_bbox[2]), int(track_bbox[3])),
                          (255, 255, 255), 2)
            cv2.putText(frame, str(track_id), (int(track_bbox[0]), int(track_bbox[1])), 0, 5e-3 * 200,
                        (0, 255, 0), 2)

        cv2.imshow('', frame)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
