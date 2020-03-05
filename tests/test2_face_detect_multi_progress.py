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

SHOW_FLAG = True

def face_detect(cameraID, camera_address, camera_rotate):
    import cv2
    import time

    from persons_track.utils.camera_capture import VideoCapture
    from persons_track.face_identity_use_baidu import baidu_face_detect
    # 窗口设置
    if SHOW_FLAG:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    video_capture = VideoCapture(cameraID, camera_address, camera_rotate)

    while True:
        start = time.time()
        cameraID, frame_i, frame = video_capture.read()
        if frame is None:
            print("get frame timeout")
            continue
        print("Frame_i", frame_i)
        print("读图片时间: ", time.time() - start)

        # 人脸检测
        start = time.time()
        face_detect_data_list = baidu_face_detect(frame)
        for face in face_detect_data_list:
            # print(face.keys()) # dict_keys(['angle', 'face_quality', 'face_token', 'face_box', 'age', 'gender', 'face_img'])
            face_box = face['face_box']
            age = face['age']
            gender = face['gender']
            print("年龄: ",age, ", 性别: ", gender)
            cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (255, 0, 255), 2)
        print("人脸检测时间: ",time.time() - start)
        if SHOW_FLAG:
            cv2.imshow('test', frame)
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


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

    cameraID = 105

    # camera相关信息和数据
    camera_info_and_data = CameraInfoAndData(CAMERAS_INFO[cameraID])
    camera_address = camera_info_and_data.url
    camera_frame_rate = camera_info_and_data.frame_rate
    camera_rotate = camera_info_and_data.rotate
    # camera_size = camera_info_and_data.size
    # area_info_list = camera_info_and_data.area_info_list
    # frame_data_queue = camera_info_and_data.frame_data_queue  # 图像帧队列

    processes = [mp.Process(target=face_detect, args=(cameraID, camera_address, camera_rotate))  # 人脸检测进程
                 ]

    [process.start() for process in processes]
    [process.join() for process in processes]
