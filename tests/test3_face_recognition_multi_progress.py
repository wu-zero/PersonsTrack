__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

# 当前文件夹的母文件夹加入系统路径
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置环境变量
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

SHOW_FLAG = False


def face_detect_and_identify_use_pool(cameraID, camera_address, camera_rotate, recognition_result_queue):
    import cv2
    import time
    import multiprocessing as mp
    from persons_track.utils.camera_capture import VideoCapture
    from persons_track.face_identity_use_baidu import face_detect_and_identify

    video_capture = VideoCapture(cameraID, camera_address, camera_rotate)

    pool = mp.Pool(processes=20)
    LAST_ADD_TIME = mp.Manager().Value('d', 0)
    LAST_ADD_TIME_LOCK = mp.Manager().Lock()
    # 窗口设置
    if SHOW_FLAG:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    while True:
        start = time.time()
        cameraID, frame_i, frame = video_capture.read()
        if frame is None:
            print("get frame timeout")
            continue
        print("In frame_i", frame_i)
        print("读图片时间: ", time.time() - start)
        pool.apply_async(face_detect_and_identify, args=(cameraID, frame_i, frame, recognition_result_queue, LAST_ADD_TIME, LAST_ADD_TIME_LOCK))

        cv2.putText(frame, str(frame_i), (int(50), int(50)), 0, 5e-3 * 400, (0, 255, 0), 3)
        if SHOW_FLAG:
            cv2.imshow('test', frame)
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def result_show(recognition_result_queue):
    while True:
        _, cameraID, frame_i, result = recognition_result_queue.get()

        print(cameraID, frame_i, result)


if __name__ == '__main__':
    import multiprocessing as mp

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

    cameraID = 106

    # camera相关信息和数据
    camera_info_and_data = CameraInfoAndData(CAMERAS_INFO[cameraID])
    camera_address = camera_info_and_data.url
    camera_frame_rate = camera_info_and_data.frame_rate
    camera_rotate = camera_info_and_data.rotate
    camera_size = camera_info_and_data.size
    area_info_list = camera_info_and_data.area_info_list
    frame_data_queue = camera_info_and_data.frame_data_queue  # 图像帧队列
    recognition_result_queue = camera_info_and_data.all_result_queue  # 人脸识别结果队列

    processes = [mp.Process(target=face_detect_and_identify_use_pool, args=(cameraID, camera_address, camera_rotate, recognition_result_queue)),
                 mp.Process(target=result_show, args=(recognition_result_queue,))]

    [process.start() for process in processes]
    [process.join() for process in processes]
