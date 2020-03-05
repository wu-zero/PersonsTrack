__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import multiprocessing as mp
import ast
# mp.set_start_method(method='spawn', force=True)  # init


class CameraInfoAndData:
    """
    摄像头信息和数据
    信息：只读属性
    数据：进程间通信队列
    """
    def __init__(self, camera_info):
        # 摄像头相关信息  url size area_info_list
        ID = camera_info['cameraID']
        url = camera_info['url']
        frame_rate = camera_info['frame_rate']
        rotate = camera_info['rotate']
        size = self.parse_size(camera_info['size'])

        self.__ID = ID
        self.__url = url
        self.__frame_rate = frame_rate
        self.__rotate = rotate
        self.__size = size

        area_info_list = []
        try:
            area_info_list = self.parse_area_info_list(camera_info['area_info_list'])
        except Exception as er:
            print('cameraID: ' + str(ID)+' area_info_list err')
        finally:
            self.__area_info_list = area_info_list

        # 数据队列
        self.frame_data_queue = mp.Queue(3)
        self.track_gpu_calculate_result_queue = mp.Queue(3)
        self.all_result_queue = mp.Manager().Queue(3)

    def __str__(self):
        result = ""
        result += "ID " + str(self.__ID) + "\n"
        result += "url " + str(self.__url) + "\n"
        result += "frame_rate " + str(self.__frame_rate) + "\n"
        result += "rotate " + str(self.__rotate) + "\n"
        result += "size " + str(self.__size) + "\n"
        result += "area_info_list " + str(self.__area_info_list) + "\n"

        return result

    @staticmethod
    def parse_size(size_str):
        num1, num2 = size_str.split('x')
        return [int(num1), int(num2)]
    @staticmethod
    def parse_area_info_list(area_info_list_str):
        area_info_list = ast.literal_eval(area_info_list_str)
        for area_info in area_info_list:
            area_info[0] = int(area_info[0])
        return area_info_list

    @property
    def ID(self):
        return self.__ID

    @property
    def url(self):
        return self.__url

    @property
    def frame_rate(self):
        return self.__frame_rate

    @property
    def rotate(self):
        return self.__rotate

    @property
    def size(self):
        return self.__size

    @property
    def area_info_list(self):
        return self.__area_info_list


if __name__ == '__main__':
    from docs.conf import CAMERAS_INFO
    CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}

    cameraID_list = [119]  # 119 109 120 114 118 121
    CAMERA_NUM = len(cameraID_list)
    camera_dict = {}  # key:cameraID, value:camera相关信息和数据
    for cameraID in cameraID_list:
        camera_dict[cameraID] = CameraInfoAndData(CAMERAS_INFO[cameraID])