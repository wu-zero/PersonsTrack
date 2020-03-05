__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import numpy as np
from collections import Counter
import time


MaxHistoryNumInRAM = 15
UseAreaIDHistory = 15
StillStatusIouScore = 0.5


class Person:
    """
    顾客类，属性：faceID, age, gender, areaID等
    """
    def __init__(self, cameraID, trackID):
        # 对应的摄像头
        self.__cameraID = cameraID
        # 对应的追踪ID
        self.__trackID = trackID
        # 对应的人脸ID和历史数据
        self.__faceID = None
        self.__faceID_num_dict = {}

        # 人脸属性
        self.__age = None
        self.__age_num_dict = {}
        self.__gender = None
        self.__gender_num_dict = {}

        # 轨迹框和轨迹框的历史
        self.__body_box = None
        self.__body_box_history = []
        self.__body_box_update_num = 0

        # 区域判断
        self.__areaID = None
        self.__areaID_history = []

        # # 人是否是静止的
        # self.__still_status_flag_history = []

    @property
    def faceID(self):
        return self.__faceID

    @property
    def age(self):
        return self.__age

    @property
    def gender(self):
        return self.__gender

    @property
    def trackID(self):
        return self.__trackID

    @property
    def areaID(self):
        return self.__areaID

    @property
    def bbox(self):
        return self.__body_box

    @property
    def bbox_history(self):
        return self.__body_box_history

    def _get_data(self):
        data = {
            'cameraID': self.__cameraID,
            'trackID': self.__trackID,
            'faceID': self.__faceID,
            'age': self.__age,
            'gender': self.__gender,
            'areaID': self.__areaID,
            'track_list': self.__body_box_history
        }
        return data

    def set_faceID(self, faceID_new):
        if faceID_new in self.__faceID_num_dict:
            self.__faceID_num_dict[faceID_new] += 1
        else:
            self.__faceID_num_dict[faceID_new] = 1

        # 根据历史数据排序，找出置信度最高的faceID
        faceID_num_dict_sorted = sorted(self.__faceID_num_dict.items(), key=lambda x: x[1], reverse=True)
        faceID = faceID_num_dict_sorted[0][0]

        self.__faceID = faceID

    def set_areaID(self, areaID_new):
        self.__areaID_history.append(areaID_new)
        if len(self.__areaID_history) >= UseAreaIDHistory:
            self.__areaID_history.pop(0)
        self.__areaID = Counter(self.__areaID_history).most_common(1)[0][0]

    def set_age(self, age_new):
        if age_new in self.__age_num_dict:
            self.__age_num_dict[age_new] += 1
        else:
            self.__age_num_dict[age_new] = 1

        # 根据历史数据排序，找出置信度最高的faceID
        age_num_dict_sorted = sorted(self.__age_num_dict.items(), key=lambda x: x[1], reverse=True)
        age = age_num_dict_sorted[0][0]

        self.__age = age

    def set_gender(self, gender_new):
        if gender_new in self.__age_num_dict:
            self.__gender_num_dict[gender_new] += 1
        else:
            self.__gender_num_dict[gender_new] = 1

        # 根据历史数据排序，找出置信度最高的faceID
        gender_num_dict_sorted = sorted(self.__gender_num_dict.items(), key=lambda x: x[1], reverse=True)
        gender = gender_num_dict_sorted[0][0]

        self.__gender = gender

    def add_body_box(self, bbox):
        self.__body_box_update_num += 1

        if type(bbox) is np.ndarray:
            bbox = bbox.tolist()
        self.__body_box = bbox
        self.__body_box_history.append({'timestamp': round(time.time() * 1000), 'rect': bbox})
        while len(self.__body_box_history) > MaxHistoryNumInRAM:
            self.__body_box_history.pop(0)

    def get_output_data(self):
        if self.__body_box_update_num % MaxHistoryNumInRAM == 0 and self.__body_box_update_num != 0:
            result = self._get_data()
            return True, result
        else:
            return False, ""

    def person_still_status_flag(self):
        if self.__body_box_update_num <= 10:
            return False
        else:
            bbox_now = self.__body_box_history[-1]['rect']
            bbox_5_before = self.__body_box_history[-6]['rect']
            # bbox_10_before = self.__body_box_history[-11]['rect']
            if self.calculate_iou(bbox_now, bbox_5_before) > 0.4:
                return True
            else:
                return False

    @staticmethod
    def calculate_iou(rect1, rect2):
        s_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        s_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        # 计算两矩形面积和
        sum_area = s_rect1 + s_rect2

        # 找到相交矩阵
        left_line = max(rect1[1], rect2[1])
        right_line = min(rect1[3], rect2[3])
        top_line = max(rect1[0], rect2[0])
        bottom_line = min(rect1[2], rect2[2])

        # 判断是否相交
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


class PersonsManage:
    """
    单一摄像头的行人管理
    """
    def __init__(self, cameraID):
        self.__cameraID = cameraID
        # 存每一个person， key：trackID, value:person
        self.__persons_dict = {}
        # 存所有person
        self.__persons_trackID_list = []

    def __add_person(self, person):
        self.__persons_dict[person.trackID] = person
        self.__persons_trackID_list.append(person.trackID)

    def add_person_use_trackID(self, trackID):
        people_new = Person(self.__cameraID, trackID)
        self.__add_person(people_new)
        # TODO send data to db
        #  use:cameraID

    def delete_person_use_trackID(self, trackID):
        if trackID in self.__persons_trackID_list:
            self.__persons_trackID_list.remove(trackID)
        if trackID in self.__persons_dict:
            del self.__persons_dict[trackID]

    def set_person_faceID(self, trackID, faceID):
        person = self.__persons_dict[trackID]
        person.set_faceID(faceID)

    def set_person_age(self, trackID, age):
        person = self.__persons_dict[trackID]
        person.set_age(age)

    def set_person_gender(self, trackID, gender):
        person = self.__persons_dict[trackID]
        person.set_gender(gender)

    def set_person_areaID(self, trackID, areaID):
        person = self.__persons_dict[trackID]
        person.set_areaID(areaID)

    def update_person_body_box(self, trackID, bbox):
        person = self.__persons_dict[trackID]
        person.add_body_box(bbox)

    def get_person_bbox(self, trackID):
        person = self.__persons_dict[trackID]
        return person.bbox

    def get_person_output_data(self, trackID):
        person = self.__persons_dict[trackID]
        return person.get_output_data()

    def get_person_still_status_flag(self, trackId):
        person = self.__persons_dict[trackId]
        return person.person_still_status_flag()

    @property
    def persons_list(self):
        return self.__persons_trackID_list



if __name__ == '__main__':
    def test_person_class():
        person = Person('camera1', 1)
        # 测试set_faceID=====================================
        print("测试set_faceID")
        person.set_faceID(111)
        print(person.faceID)
        person.set_faceID(222)
        print(person.faceID)
        person.set_faceID(222)
        print(person.faceID)
        person.set_faceID(111)
        print(person.faceID)
        person.set_faceID(333)
        print(person.faceID)
        person.set_faceID(333)
        print(person.faceID)
        person.set_faceID(333)
        print(person.faceID)

        # 测试set_areaID========================================
        print("测试set_areaID")
        for i in range(15):
            person.set_areaID('123')
            print(i, person.areaID)
        for i in range(15):
            person.set_areaID('456')
            print(i+15, person.areaID)

        # 测试add_bbox===========================================
        print("测试add_bbox")
        for i in range(30):
            box = [i, i, 100+i, 100+i]
            person.add_body_box(box)
            print(i, person.get_output_data())
            print(person.person_still_status_flag())

    def test_person_manage_class():
        pm = PersonsManage('camera1')
        pm.add_person_use_trackID('person1')
        pm.add_person_use_trackID('person2')
        pm.set_person_areaID('person1', 'area1')
        # pm.set_person_areaID('person1', 'area2')
        # pm.set_person_areaID('person1', 'area2')
        pm.set_person_faceID('person1', 'xiaoming')
        # pm.set_person_faceID('person1', 'xiaogang')
        # pm.set_person_faceID('person1', 'xiaogang')

        for i in range(15):
            pm.update_person_body_box('person1', [i, i, i, i])
            flag, output_data = pm.get_person_output_data('person1')
            if flag is True:
                print(i, output_data)


    test_person_class()
    # test_person_manage_class()





