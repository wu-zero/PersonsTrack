__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import numpy as np
import operator
import cv2


class Rect:
    """
    矩形类
    """
    def __init__(self, *args, **kwargs):
        self._left, self._top, self._right, self._bottom = self.get_rect_x0_y0_x1_y1_from_various_input(*args)
        self._square = None

    def top_part(self, part_proportion=0.5):
        assert 0 < part_proportion <= 1
        return Rect(self._left, self._top, self._right, self._top + (self._bottom - self._top) * part_proportion)

    @property
    def left(self):
        return self._left

    @property
    def top(self):
        return self._top

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    @property
    def square(self):
        if self._square:
            return self._square
        else:
            self._square = (self._right - self._left) * (self._bottom - self._top)
            # print(self._square)
            return self._square

    @staticmethod
    def get_rect_x0_y0_x1_y1_from_various_input(*args):
        """
        #  (x0,y0)
        #        ---------
        #        |       |
        #        |       |
        #        |       |
        #        |       |
        #        ---------
        #             (x1,y1)
        :param args: 输入,list或np.ndarray
        :return:矩形的两个顶点坐标
        """
        try:
            # print(args)
            if len(args) == 4:
                return int(args[0]), int(args[1]), int(args[2]), int(args[3])
            elif len(args) == 1:
                rect_data = args[0]
                if isinstance(rect_data, list):
                    if len(rect_data) == 4:
                        return int(rect_data[0]), int(rect_data[1]), int(rect_data[2]), int(rect_data[3])
                    elif len(rect_data) == 2:
                        return int(rect_data[0][0]), int(rect_data[0][1]), int(rect_data[1][0]), int(rect_data[1][1])
                elif isinstance(rect_data, np.ndarray):
                    if operator.eq(rect_data.shape, (4,)):
                        return int(rect_data[0]), int(rect_data[1]), int(rect_data[2]), int(rect_data[3])
                    elif operator.eq(rect_data.shape, (2, 2)):
                        return int(rect_data[0][0]), int(rect_data[0][1]), int(rect_data[1][0]), int(rect_data[1][1])
                else:
                    raise ValueError("Input error")
            else:
                raise ValueError("Input error")
        except Exception as err:
            raise ValueError("Input error")

    def __str__(self):
        return str([self._left, self._top, self._right, self._bottom])


class Point:
    """
    点类
    """
    def __init__(self, *args, **kwargs):
        self._x, self._y = self.get_point_x_y_from_various_input(*args)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @staticmethod
    def get_point_x_y_from_various_input(*args):
        try:
            if len(args) == 2:
                return int(args[0]), int(args[1])
            elif len(args) == 1:
                point_data = args[0]
                if isinstance(point_data, list):
                    return int(point_data[0]), int(point_data[1])
                elif isinstance(point_data, np.ndarray):
                    return int(point_data[0]), int(point_data[1])
                else:
                    raise ValueError("Input error")
            else:
                raise ValueError("Input error")
        except Exception as err:
            raise ValueError("Input error")


class Area:
    """
    区域类，用于顾客位置判断
    """
    def __init__(self, img_shape, areaID, vertex_points, min_body_box_square=0):
        self._areaID = areaID
        self._img_shape = None
        self._vertex_points = None
        self._area = None
        self._min_body_box_square = min_body_box_square

        if 2 <= len(img_shape) <= 3:
            self._img_shape = (img_shape[0], img_shape[1])
        else:
            print('wrong img_shape')

        if isinstance(vertex_points, list):
            vertex_points_array = np.array(vertex_points)
        else:
            vertex_points_array = vertex_points

        if len(vertex_points_array.shape) == 2 and vertex_points_array.shape[1] == 2:
            self._vertex_points = vertex_points_array
        else:
            print('wrong vertex_shape')

        img = np.zeros(img_shape, np.uint8)
        cv2.fillConvexPoly(img, self._vertex_points, 1)
        self._area = img

    @property
    def areaID(self):
        return self._areaID

    @property
    def area_img(self):
        return self._area * 255

    def draw_area_to_img(self, img):
        cv2.polylines(img, [self._vertex_points.reshape((-1, 1, 2))], True, (255, 255, 0), thickness=3)
        cv2.putText(img, str(self._areaID), tuple(self._vertex_points[0]), 0, 5e-3 * 200, (255, 255, 0), 2)

    def _rect_in_area_square(self, rect):
        row0, row1 = rect.top, rect.bottom
        col1, col2 = rect.left, rect.right
        return self._area[row0:row1, col1:col2].sum()

    def _rect_in_area_rate(self, rect):
        rect_square = rect.square
        rect_in_area_square = self._rect_in_area_square(rect)
        rate = rect_in_area_square/rect_square
        assert 0 <= rate <= 1
        return rate

    def _rect_not_in_area_rate(self, rect):
        rect_in_area_square = self._rect_in_area_rate(rect)
        rate = 1 - rect_in_area_square
        return rate

    def _point_in_area(self, point):
        row, col = point.y, point.x
        return self._area[row, col] == 1

    def face_box_in_area(self, face_box):
        if self._rect_in_area_rate(face_box) > 0.9:
            return True
        else:
            return False

    def body_box_in_area_score(self, body_box):
        """
        身体框足够大，在目标区域中的面积所占比例足够高
        :param body_box:
        :return:
        """
        if body_box.square < self._min_body_box_square:
            return 0
        else:
            use_box_1 = body_box
            use_box_2 = body_box.top_part(0.8)
            if self._rect_in_area_square(use_box_1) > self._min_body_box_square:
                use_box_1_score = self._rect_in_area_rate(use_box_1)
                use_box_2_score = self._rect_in_area_rate(use_box_2)
            else:
                return 0

        return max(use_box_1_score, use_box_2_score)

    def point_in_area_score(self, point):
        if point is None:
            return 0
        else:
            if self._point_in_area(point):
                return 1
            else:
                return 0

    def points_in_area_score(self, point_list):
        points_num = len(point_list)
        if points_num == 0:
            return 0
        score = 0
        for point in point_list:
            score += self.point_in_area_score(point)

        score = score/points_num
        return score


class AreaJudge:
    def __init__(self, img_shape, area_info_list, mode=1):
        self._mode = mode
        self._area_list = []
        self._area_num = len(area_info_list)
        for area_info in area_info_list:
            areaID, area_vertex, area_type = area_info
            self._area_list.append(Area(img_shape, areaID, area_vertex, area_type))

    def draw(self, frame):
        for area in self._area_list:
            area.draw_area_to_img(frame)

    def judge(self, body_data):
        if self._area_num == 0:
            return None
        if self._mode == 1:
            body_box = body_data['body_box']
            if not isinstance(body_box, Rect):
                body_box = Rect(body_box)

            score_list = np.zeros(self._area_num)
            for i, area in enumerate(self._area_list):
                score_list[i] = area.body_box_in_area_score(body_box)
            result = int(np.argmax(score_list))
            return self._area_list[result].areaID if score_list[result] > 0.8 else None
        elif self._mode == 2:
            body_box = body_data['body_box']
            other_data = body_data['other_data']
            nose_point = other_data['nose_point']
            key_point_list = [other_data['rhip_point'], other_data['lhip_point'], other_data['rshoulder_point'], other_data['lshoulder_point']]

            body_box = Rect(body_box)
            nose_point = Point(nose_point) if nose_point is not None else None
            key_point_list = [Point(key_point) if key_point is not None else None
                              for key_point in key_point_list]

            score_list = np.zeros(self._area_num)
            for i, area in enumerate(self._area_list):
                score1 = area.body_box_in_area_score(body_box)
                score2 = area.point_in_area_score(nose_point)
                score3 = area.points_in_area_score(key_point_list)
                if score2 == 1 and score3 >= 0.75:
                    score = (score1 + score3)/2
                else:
                    score = 0
                score_list[i] = score

            result = int(np.argmax(score_list))
            return self._area_list[result].areaID if score_list[result] > 0.8 else None
        else:
            return None


# 测试Rect类和Area类
def rect_and_area_test_main():
    area = Area(img_shape=(1080, 1920, 1), areaID='area_x',
                vertex_points=[[200, 400], [1800, 400], [1800, 600], [200, 600]])
    cv2.imshow(' ', area.area_img)
    cv2.waitKey(0)

    rect1 = Rect(1000, 400, 1800, 600)
    rect2 = rect1.top_part(0.5)
    point1 = Point(200, 400)
    print(rect1)
    print(rect2)

    print(area._rect_not_in_area_rate(rect1))
    print(area._rect_not_in_area_rate(rect2))
    print(area._point_in_area(point1))

if __name__ == '__main__':
    rect_and_area_test_main()



