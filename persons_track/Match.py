__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"


import numpy as np
from abc import abstractmethod
from scipy.optimize import linear_sum_assignment
import pprint


def _calculate_face_box_in_body_box_rate(face_box, body_box):
    """
    计算脸部框在身体框里部分的比例

    :param face_box:
    :param body_box:
    :return:
    """
    # 找到相交矩阵
    left_line = max(face_box[0], body_box[0])
    right_line = min(face_box[2], body_box[2])
    top_line = max(face_box[1], body_box[1])
    bottom_line = min(face_box[3], body_box[3])

    # 判断是否相交
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        # 人脸框面积
        S_face_rect = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        # 相交矩阵面积
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect/S_face_rect


def _calculate_face_box_body_box_distance(face_box, body_box):
    """
    计算脸部框上边界中点和在身体框上边界中点的相对距离，越小置信度越高
    :param face_box:
    :param body_box:
    :return:
    """
    face_top_center = np.array([(face_box[0] + face_box[2]) / 2, face_box[1]])
    body_top_center = np.array([(body_box[0] + body_box[2]) / 2, body_box[1]])
    distance = np.sqrt(np.sum(np.square(face_top_center-body_top_center)))
    face_rect_w_and_h = face_box[2] - face_box[0] + face_box[3] - face_box[1]
    return distance/face_rect_w_and_h


def _calculate_face_box_nose_point_distance(face_box, nose_point):
    """
    :param face_box:
    :param nose_point:
    :return:
    """
    face_center = np.array([(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2])
    nose_point = np.array(nose_point)
    distance = np.sqrt(np.sum(np.square(face_center - nose_point)))
    face_rect_w_and_h = face_box[2] - face_box[0] + face_box[3] - face_box[1]
    return distance / face_rect_w_and_h


class MatchAlgorithm:
    """
    人脸和人体匹配的算法部分
    """
    def __init__(self, max_distance):
        self._max_distance = max_distance

    @abstractmethod
    def cost_metric_fun(self, bodies_data, faces_data):
        pass

    def run(self, bodies_data, faces_data):
        # 没有要匹配的数据
        if len(bodies_data) == 0 or len(faces_data) == 0:
            return []
        # 计算cost matrix
        cost_matrix = self.cost_metric_fun(bodies_data, faces_data)
        # 最小权重匹配（匈牙利算法）
        indices = linear_sum_assignment(cost_matrix)  # The pairs of (row, col) indices in the original array giving
        result = []
        for row, col in zip(indices[0], indices[1]):
            if cost_matrix[row, col] < self._max_distance:
                result.append([row, col])
            else:
                pass
        return result


class MatchAlgorithmUseBox(MatchAlgorithm):
    def __init__(self, max_distance=1):
        MatchAlgorithm.__init__(self, max_distance)

    def cost_metric_fun(self, bodies_data, faces_data):
        body_box_list = [body_data['body_box'] for body_data in bodies_data]
        face_box_list = [face_data['face_box'] for face_data in faces_data]

        cost_matrix = np.ones((len(body_box_list), len(face_box_list))) * 255
        for row, body_box in enumerate(body_box_list):
            for col, face_box in enumerate(face_box_list):
                score1 = _calculate_face_box_in_body_box_rate(face_box, body_box)
                if score1 < 0.9:
                    pass
                else:
                    score2 = _calculate_face_box_body_box_distance(face_box, body_box)
                    cost_matrix[row, col] = score2
        return cost_matrix


class MatchAlgorithmUseBoxAndKeyPoint(MatchAlgorithm):
    def __init__(self, max_distance=1):
        MatchAlgorithm.__init__(self, max_distance)

    def cost_metric_fun(self, bodies_data, faces_data):
        body_box_list = [body_data['body_box'] for body_data in bodies_data]
        body_nose_point_list = [body_data['other_data']['nose_point'] for body_data in bodies_data]

        face_box_list = [face_data['face_box'] for face_data in faces_data]

        cost_matrix = np.ones((len(body_box_list), len(face_box_list))) * 255
        for row, (body_box, body_nose_point) in enumerate(zip(body_box_list, body_nose_point_list)):
            for col, face_box in enumerate(face_box_list):
                score1 = _calculate_face_box_in_body_box_rate(face_box, body_box)
                if score1 < 0.9:
                    pass
                else:
                    if body_nose_point is None:
                        pass
                    else:
                        score2 = _calculate_face_box_nose_point_distance(face_box, body_nose_point)
                        cost_matrix[row, col] = score2
        return cost_matrix


_Match_Algorithm_Dict = {1: MatchAlgorithmUseBox(),
                         2: MatchAlgorithmUseBoxAndKeyPoint()}


class Match:
    """
    用于脸部框和身体框的对应，从而获取跟踪目标身份信息
    """

    _MAX_BODY_FRAME = 15
    _MAX_FACE_FRAME = 7

    def __init__(self, mode=1):
        self._mode = mode
        self._face_body_match_algorithm = _Match_Algorithm_Dict[self._mode]
        
        self._frame_i_list_for_bodies_data = []
        self._frame_i_list_for_faces_data = []
        self._frames_bodies_data_dict = {}
        self._frames_faces_data_dict = {}

    def _add_one_frame_bodies_data(self, frame_i, one_frame_bodies_data):
        """
        添加身体框数据，并维护好队列
        :param frame_i: 帧数，用于与脸部框数据的对应
        :param one_frame_bodies_data: 要添加的身体框数据
        :return:
        """
        # 维护队列
        if len(self._frame_i_list_for_bodies_data) >= self._MAX_BODY_FRAME:
            frame_i_pop = self._frame_i_list_for_bodies_data.pop(0)
            frame_data_pop = self._frames_bodies_data_dict.pop(frame_i_pop)
        # 添加数据
        self._frame_i_list_for_bodies_data.append(frame_i)
        self._frames_bodies_data_dict[frame_i] = one_frame_bodies_data

    def _get_one_frame_bodies_data(self, frame_i):
        """
        根据帧数，返回身体框数据
        :param frame_i:
        :return:
        """
        if frame_i in self._frame_i_list_for_bodies_data:
            self._frame_i_list_for_bodies_data.remove(frame_i)
            frame_data_pop = self._frames_bodies_data_dict.pop(frame_i)
            return True, frame_data_pop
        else:
            return False, None

    def _add_one_frame_faces_data(self, frame_i, one_frame_faces_data):
        """
        添加脸部框数据，并维护好队列
        :param frame_i: 帧数，用于与身体框数据的对应
        :param one_frame_faces_data: 要添加的身体框数据
        :return:
        """
        # 删除过期数据
        if len(self._frame_i_list_for_faces_data) >= self._MAX_FACE_FRAME:
            frame_i_pop = self._frame_i_list_for_faces_data.pop(0)
            frame_faces_data_pop = self._frames_faces_data_dict.pop(frame_i_pop)

        self._frame_i_list_for_faces_data.append(frame_i)
        self._frames_faces_data_dict[frame_i] = one_frame_faces_data

    def _get_one_frame_faces_data(self, frame_i):
        """
        根据帧数，返回脸部框数据
        :param frame_i:
        :return:
        """
        if frame_i in self._frame_i_list_for_faces_data:
            self._frame_i_list_for_faces_data.remove(frame_i)
            frame_data_pop = self._frames_faces_data_dict.pop(frame_i)
            return True, frame_data_pop
        else:
            return False, None

    def _do_match(self, one_frame_bodies_data, one_frame_faces_data):
        # print(one_frame_bodies_data)
        # print(one_frame_faces_data)
        trackID_list = [body_data['trackID'] for body_data in one_frame_bodies_data]
        face_data_list = one_frame_faces_data

        body_idx_face_idx_pair_list = self._face_body_match_algorithm.run(one_frame_bodies_data, one_frame_faces_data)
        # print(body_idx_face_idx_pair_list)
        trackID_face_data_pair_list = []
        for body_idx_face_idx_pair in body_idx_face_idx_pair_list:
            body_idx, face_idx = body_idx_face_idx_pair
            trackID_face_data_pair_list.append([trackID_list[body_idx], face_data_list[face_idx]])
        return trackID_face_data_pair_list

    def body_frame_update(self, frame_i, bodies_data):
        flag, face_data = self._get_one_frame_faces_data(frame_i)
        # 读取到了对应帧数的脸部框数据
        if flag:
            return self._do_match(bodies_data, face_data)
        # 没有读取到对应帧数的脸部框数据
        else:
            self._add_one_frame_bodies_data(frame_i, bodies_data)
            return []

    def face_frame_update(self, frame_i, faces_data):
        flag, bodies_data = self._get_one_frame_bodies_data(frame_i)
        # 读取到了对应帧数的身体框数据
        if flag:
            return self._do_match(bodies_data, faces_data)
        # 没有读取到对应帧数的身体框数据
        else:
            self._add_one_frame_faces_data(frame_i, faces_data)
            return []


def test1():
    import cv2
    from PIL import Image
    from third_party.deep_sort_yolov3.yolo import YOLO
    from persons_track.utils.face_detect import detect_face
    import os
    from docs.conf import DATA_PATH

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    yolo = YOLO()

    def detect_body(img):
        image_for_yolo = Image.fromarray(img[..., ::-1])
        body_boxes_xywh = yolo.detect_image(image_for_yolo)
        body_boxes = []
        for box in body_boxes_xywh:
            body_boxes.append([int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])])
        return body_boxes

    img_list = [cv2.imread(os.path.join(DATA_PATH, '1.jpg')),
                cv2.imread(os.path.join(DATA_PATH, '2.jpg'))]

    # 检测到人脸框
    face_boxes_list = [detect_face(img) for img in img_list]
    # 检测到人体框
    body_boxes_list = [detect_body(img) for img in img_list]
    # 根据face_boxes构造假的face_frame_data，用于测试
    face_frame_data_list = []
    for face_boxes in face_boxes_list:
        one_frame_faces_data = []
        for i, face_box in enumerate(face_boxes):
            one_frame_faces_data.append({'faceID': 'face' + str(i), 'face_box': face_box})
        face_frame_data_list.append(one_frame_faces_data)

    # 根据body_boxes构造假的body_frame_data，用于测试
    body_frame_data_list = []
    for body_boxes in body_boxes_list:
        one_frame_bodies_data = []
        for i, body_box in enumerate(body_boxes):
            one_frame_bodies_data.append({'trackID': 'track' + str(i), 'body_box': body_box})
        body_frame_data_list.append(one_frame_bodies_data)

    print('face_frame_data_list')
    pprint.pprint(face_frame_data_list)
    print('body_frame_data_list')
    pprint.pprint(body_frame_data_list)

    # 画出face_box 和 body_box
    for img_idx in range(len(img_list)):
        img = img_list[img_idx]
        one_frame_faces_data = face_frame_data_list[img_idx]
        one_frame_bodies_data = body_frame_data_list[img_idx]
        for face_data in one_frame_faces_data:
            faceID = face_data['faceID']
            face_box = face_data['face_box']
            box = face_box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(img, faceID, (int(box[0]), int(box[1] - 3)), 0, 0.6, (255, 0, 0), 1)

        for body_data in one_frame_bodies_data:
            trackID = body_data['trackID']
            body_box = body_data['body_box']
            box = body_box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, trackID, (int(box[0]), int(box[1] - 3)), 0, 0.6, (0, 255, 0), 1)
        cv2.imshow("image" + str(img_idx), img)

    match1 = Match(mode=1)
    pprint.pprint(match1.body_frame_update(0, body_frame_data_list[0]))
    pprint.pprint(match1.body_frame_update(1, body_frame_data_list[1]))
    pprint.pprint(match1.face_frame_update(1, face_frame_data_list[1]))
    pprint.pprint(match1.face_frame_update(0, face_frame_data_list[0]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test2():
    import cv2
    from third_party.tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
    from third_party.tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh
    from persons_track.utils.face_detect import detect_face
    import os
    from docs.conf import DATA_PATH

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    model = 'cmu'
    resolution = '656x368'  # Recommends : 432x368 or 656x368 or 1312x736'
    model_w, model_h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(model_w, model_h))

    def detect_body(img):
        img_h, img_w = img.shape[0:2]
        humans = e.inference(img, resize_to_default=(model_w > 0 and model_h > 0), upsample_size=4.0)
        body_result = []
        for human in humans:
            result = human.get_useful_data(img_w, img_h, img_w, img_h)
            if result:
                body_result.append(result)
        return body_result

    img_list = [cv2.imread(os.path.join(DATA_PATH, '1.jpg')),
                cv2.imread(os.path.join(DATA_PATH, '2.jpg'))]

    # 检测到人脸框
    face_boxes_list = [detect_face(img) for img in img_list]
    # 根据face_boxes构造假的face_frame_data，用于测试
    face_frame_data_list = []
    for face_boxes in face_boxes_list:
        one_frame_faces_data = []
        for i, face_box in enumerate(face_boxes):
            one_frame_faces_data.append({'faceID': 'face' + str(i), 'face_box': face_box})
        face_frame_data_list.append(one_frame_faces_data)

    # 检测到人体信息
    body_data_list = [detect_body(img) for img in img_list]
    # 根据body_boxes构造假的body_frame_data，用于测试
    body_frame_data_list = []
    for body_data in body_data_list:
        one_frame_bodies_data = []
        for i, one_body_data in enumerate(body_data):
            one_frame_one_body_data = {}
            one_frame_one_body_data['trackID'] = 'track' + str(i)
            box = one_body_data['body_box'].copy()
            other_data = one_body_data['other_data'].copy()
            one_frame_one_body_data['body_box'] = [int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])]
            one_frame_one_body_data['other_data'] = other_data
            one_frame_bodies_data.append(one_frame_one_body_data)
        body_frame_data_list.append(one_frame_bodies_data)

    print('face_frame_data_list')
    pprint.pprint(face_frame_data_list)
    print('body_frame_data_list')
    pprint.pprint(body_frame_data_list)

    # 画出face_box 和 body_box
    for img_idx in range(len(img_list)):
        img = img_list[img_idx]
        one_frame_faces_data = face_frame_data_list[img_idx]
        one_frame_bodies_data = body_frame_data_list[img_idx]
        for face_data in one_frame_faces_data:
            faceID = face_data['faceID']
            face_box = face_data['face_box']
            box = face_box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(img, faceID, (int(box[0]), int(box[1] - 3)), 0, 0.6, (255, 0, 0), 1)

        for body_data in one_frame_bodies_data:
            trackID = body_data['trackID']
            body_box = body_data['body_box']
            nose_point = body_data['other_data']['nose_point']
            box = body_box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.circle(img, tuple(nose_point), 5, (0, 255, 0), thickness=-1)
            cv2.putText(img, trackID, (int(box[0]), int(box[1] - 3)), 0, 0.6, (0, 255, 0), 1)
        cv2.imshow("image" + str(img_idx), img)

    match2 = Match(mode=2)
    pprint.pprint(match2.body_frame_update(0, body_frame_data_list[0]))
    pprint.pprint(match2.body_frame_update(1, body_frame_data_list[1]))
    pprint.pprint(match2.face_frame_update(1, face_frame_data_list[1]))
    pprint.pprint(match2.face_frame_update(0, face_frame_data_list[0]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test2()
