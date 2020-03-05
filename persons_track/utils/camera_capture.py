__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import cv2
import time
import threading
from queue import Queue

DebugFlag = False


class VideoCapture:
    def __init__(self, cameraID, camera_url, rotate=0):
        self._cameraID = cameraID
        self._camera_url = camera_url
        self._rotate = rotate
        self._is_video = False
        self._video_totalFrame = 0
        self._frame_i = 0

        # 判断是不是本地视频
        if isinstance(camera_url, str) and camera_url[-3:] == 'mp4':
            self._is_video = True
        # 打开视频流
        self._cap = cv2.VideoCapture(self._camera_url)
        # 输出本地视频相关参数
        if self._is_video:
            print("原始帧率", self._cap.get(cv2.CAP_PROP_FPS))
            print("原始宽", self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print("原始高", self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("原始帧数", self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_totalFrame = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) * 12
        # 设置循环读取线程
        self.image_q = Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        read_count = 0
        fail_count = 0
        while True:
            start = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if not self._cap.isOpened():
                    print("camera", self._cameraID, "closed, retry in 5 sec")
                    time.sleep(5)
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._camera_url)
                else:
                    print("camera", self._cameraID, "return null")
                    fail_count += 1
                    time.sleep(0.04)
                    if fail_count > 20:
                        self._cap.release()
                # !!!!!!!!!
                continue
            else:
                read_count += 1
                fail_count = 0

            if self._is_video is True:
                time.sleep(0.03)
                if read_count % self._video_totalFrame == self._video_totalFrame - 1:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if not self.image_q.empty():
                try:
                    self.image_q.get_nowait()   # discard previous (unprocessed) frame

                except Exception as err:
                    pass
            self.image_q.put(frame)
            if DebugFlag:
                print( "camraID: ", self._cameraID, " 读网络帧时间：", time.time()-start)

    def read(self):
        self._frame_i += 1
        frame = None
        try:
            frame = self.image_q.get(block=True, timeout=0.01)
            start = time.time()
            if self._rotate in [-1, 1]:
                # frame = np.rot90(frame, self._rotate)
                if self._rotate == -1:
                    frame = cv2.transpose(frame)
                    frame = cv2.flip(frame, 1)
                if self._rotate == 1:
                    frame = cv2.transpose(frame)
                    frame = cv2.flip(frame, 0)
            # print('rotate time',time.time()-start)
        except Exception as err:
            print("CameraID: ", self._cameraID, " 获取数据出错")
        return [self._cameraID, self._frame_i, frame]

def camera_read(cameraID, camera_url, frame_data_queue, rotate=0, time_per_frame=0.2):
    video = VideoCapture(cameraID, camera_url, rotate=rotate)
    frame_i = 0
    while True:
        frame_i += 1
        start = time.time()
        frame_data = video.read()

        if not frame_data_queue.empty():
            try:
                frame_data_queue.get_nowait()  # discard previous (unprocessed) frame
            except Exception as err:
                pass

        frame_data_queue.put(frame_data)
        use_time = time.time() - start
        if DebugFlag:
            print("从线程队列里获取帧时间：", use_time)
        if use_time > time_per_frame:
             pass
        else:
            time.sleep(time_per_frame-use_time)


def img_show(frames_data_queue):
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    while True:
        try:
            start = time.time()
            frame_data = frames_data_queue.get()
            cameraID, frame_i, frame = frame_data
            if DebugFlag:
                print("从进程队列里获取帧时间：", time.time()-start)
            print(frame_i)
            time.sleep(0.2)
            # if frame is not None:
            #     cv2.imshow("test", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
        except Exception as err:
            print('ImgShow', err)


if __name__ == '__main__':
    from persons_track.CameraInfoAndData import CameraInfoAndData

    from docs.conf import TEST_FLAG
    if TEST_FLAG:
        from docs.conf import CAMERAS_INFO

        CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}
    else:
        from persons_track.utils.modules.dbutil import MySQLPlugin
        from docs.conf import DB_CONF

        db = MySQLPlugin(host=DB_CONF['server'], user=DB_CONF['user'], password=DB_CONF['pwd'], db=DB_CONF["db"])
        CAMERAS_INFO = db.query_camera_data()  # 从数据库获取摄像头info
        CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}

    cameraID = 108

    # camera相关信息和数据
    camera_info_and_data = CameraInfoAndData(CAMERAS_INFO[cameraID])
    camera_address = camera_info_and_data.url
    camera_frame_rate = camera_info_and_data.frame_rate
    camera_rotate = camera_info_and_data.rotate
    camera_size = camera_info_and_data.size
    area_info_list = camera_info_and_data.area_info_list
    frame_data_queue = camera_info_and_data.frame_data_queue  # 图像帧队列

    # test_video_capture(cameraID, camera_address, camera_rotate)

    import multiprocessing as mp
    processes = [mp.Process(target=camera_read, args=(cameraID, camera_address, frame_data_queue, camera_rotate)),
                 mp.Process(target=img_show, args=(frame_data_queue,))]

    [p.start() for p in processes]
    [p.join() for p in processes]