__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"


import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_face_box_in_body_box_rate(face_rect, body_rect):
    """
    计算脸部框在身体框里的比例，越大置信度越高
    :param face_rect:
    :param body_rect:
    :return:
    """
    # 找到相交矩阵
    left_line = max(face_rect[0], body_rect[0])
    right_line = min(face_rect[2], body_rect[2])
    top_line = max(face_rect[1], body_rect[1])
    bottom_line = min(face_rect[3], body_rect[3])

    # 判断是否相交
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        # 人脸框面积
        S_face_rect = (face_rect[2] - face_rect[0]) * (face_rect[3] - face_rect[1])
        # 相交矩阵面积
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect/S_face_rect


# ============================================mode1==========================
def calculate_face_box_body_box_distance(face_rect, body_rect):
    """
    计算脸部框上边界中点和在身体框上边界中点的相对距离，越小置信度越高
    :param face_rect:
    :param body_rect:
    :return:
    """
    face_top_center = np.array([(face_rect[0]+face_rect[2])/2, face_rect[1]])
    body_top_center = np.array([(body_rect[0]+body_rect[2])/2, body_rect[1]])
    distance = np.sqrt(np.sum(np.square(face_top_center-body_top_center)))
    face_rect_w_and_h = face_rect[2]-face_rect[0]+face_rect[3]-face_rect[1]
    return distance/face_rect_w_and_h


# 用人脸框和人体框
def calculate_face_box_body_box_match_score(faces_data, bodies_data):
    face_box_list = faces_data['face_box_list']
    body_box_list = bodies_data['body_box_list']

    cost_matrix = np.ones((len(body_box_list), len(face_box_list))) * 255
    for row, body_box in enumerate(body_box_list):
        for col, face_box in enumerate(face_box_list):
            score1 = calculate_face_box_in_body_box_rate(face_box, body_box)
            if score1 < 0.9:
                pass
            else:
                score2 = calculate_face_box_body_box_distance(face_box, body_box)
                cost_matrix[row, col] = score2
    return cost_matrix


# ============================================mode2==========================
def calculate_face_box_nose_point_distance(face_rect, nose_point):
    """
    :param face_rect:
    :param nose_point:
    :return:
    """
    face_center = np.array([(face_rect[0]+face_rect[2])/2, (face_rect[1]+face_rect[3])/2])
    nose_point = np.array(nose_point)
    distance = np.sqrt(np.sum(np.square(face_center - nose_point)))
    face_rect_w_and_h = face_rect[2] - face_rect[0] + face_rect[3] - face_rect[1]
    return distance / face_rect_w_and_h


# 用人脸框和人体框、关键点
def calculate_face_box_body_box_match_score2(faces_data, bodies_data):
    face_box_list = faces_data['face_box_list']
    body_box_list = bodies_data['body_box_list']
    body_nose_point_list = bodies_data['body_nose_point_list']

    cost_matrix = np.ones((len(body_box_list), len(face_box_list))) * 255
    for row, (body_box, body_nose_point) in enumerate(zip(body_box_list, body_nose_point_list)):
        for col, face_box in enumerate(face_box_list):
            score1 = calculate_face_box_in_body_box_rate(face_box, body_box)
            if score1 < 0.9:
                pass
            else:
                score2 = calculate_face_box_nose_point_distance(face_box, body_nose_point)
                cost_matrix[row, col] = score2
    return cost_matrix


# ============================================================================
#
def do_match(distance_metric_fun, bodies_data, faces_data):
    """
    匹配身体框和脸部框
    :param distance_metric_fun:
    :param bodies_data:
    :param faces_data:
    :return: [[身体框编号，脸部框编号]...]
    """
    MAX_DISTANCE = 0.8

    if len(bodies_data) == 0 or len(faces_data) == 0:
        return []  # Nothing to match.
    cost_matrix = distance_metric_fun(bodies_data, faces_data)

    indices = linear_sum_assignment(cost_matrix)  # The pairs of (row, col) indices in the original array giving
    result = []
    for row, col in zip(indices[0], indices[1]):
        if cost_matrix[row, col] < MAX_DISTANCE:
            result.append([row, col])
        else:
            pass
    return result

if __name__ == '__main__':
    face_rect1 = [0, 0, 2, 2]
    body_rect1 = [0, 0, 4, 4]
    print(calculate_face_box_in_body_box_rate(face_rect1, body_rect1))
    print(calculate_face_box_body_box_distance(face_rect1, body_rect1))


