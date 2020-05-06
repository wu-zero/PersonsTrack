import os
import sys
# 设置环境变量
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import threading
import multiprocessing as mp

import time
import math
import json
import numpy as np
import cv2

# 人体检测和追踪的第三方库（改）
from third_party.tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from third_party.tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh
from third_party.deep_sort_yolov3.deep_sort_model import DeepSortPreprocess, DeepSort
# 自己写的库
from persons_track.utils.camera_capture import VideoCapture
from persons_track.face_identity_use_baidu import face_detect_and_identify
from persons_track.PersonManage import PersonsManage
from persons_track.Match import Match
from persons_track.AreaJudge import AreaJudge
from persons_track.CameraInfoAndData import CameraInfoAndData
from persons_track.DataFusion import AreaDataFusion
from persons_track.utils.MsgSender import MsgSender
from persons_track.utils.others import resize_boxes
from persons_track.Logger import Logger

# 配置log文件相关
from docs.conf import LOGS_PATH
LOG_FLAG = False
log_path = os.path.join(LOGS_PATH, os.path.basename(sys.argv[0]).split(".")[0])
if not os.path.exists(log_path):
    os.makedirs(log_path)
log = Logger(os.path.join(log_path, 'all.log'), file_level='ERROR', terminal_flag=False, terminal_level='INFO')


IMG_ORIGINAL_W, IMG_ORIGINAL_H = 2560, 1440
IMG_FOR_GPU_W, IMG_FOR_GPU_H = 656, 368


# 放数据到队列
def put_data_to_queue(data_queue: mp.Queue, data, queue_name: str):
    # 队列满了，删除较旧的信息
    if data_queue.full():
        try:
            data_queue.get_nowait()
        except Exception as err:
            # print(queue_name, ' get old data fail', err)
            pass
        else:
            # print(queue_name, ' get old data success', err)
            pass
    # 放入数据
    try:
        data_queue.put_nowait(data)
    except Exception as err:
        log.logger.error(queue_name + " put data fail")
        # print(queue_name, " put data fail")


# 从队列中取数据
def get_data_from_queue(data_queue: mp.Queue, timeout: float, queue_name: str):
    data = None
    try:
        data = data_queue.get(block=True, timeout=timeout)
    except Exception as err:
        log.logger.error(queue_name + " get data fail")
        # print(queue_name, " get data fail")
    return data


# 多线程读取多摄像头视频帧、数据分发
def frame_data_get_and_distribute(camera_dict, track_frame_data_queue):
    """
    从frame_data_queue获取图像帧并进行数据分发， 分发给track进程和identify进程
    :param camera_dict: 摄像头信息和数据， 用到：ID、 frame_data_queue
    :param track_frame_data_queue: 用于track的队列列表， 每个GPU对应一个对列
    :return:
    """
    # 摄像头数目，gpu数目
    camera_num = len(camera_dict)
    print("Camera num: ", camera_num)
    camera_id_list = list(camera_dict.keys())
    video_cap_dict = {camera_id: VideoCapture(camera_id, camera_dict[camera_id].url, camera_dict[camera_id].rotate)
                      for camera_id in camera_id_list}
    identify_result_queue_dict = {camera_id: camera_dict[camera_id].all_result_queue
                                  for camera_id in camera_id_list}
    pool = mp.pool.ThreadPool(20)
    while True:
        # 用线程池获取frame_data
        start = time.time()
        # 每个GPU对应一个frame_data_for_body_queue
        for camera_id in camera_id_list:
            frame_data = video_cap_dict[camera_id].read()  # 读camera数据
            if isinstance(frame_data, list) and len(frame_data) == 3 and frame_data[2] is not None:
                cameraID, frame_i, frame = frame_data
                # 1 identify 用线程池进行人脸检测和识别,放入结果队列
                identify_result_queue = identify_result_queue_dict[cameraID]
                pool.apply_async(face_detect_and_identify, args=(cameraID, frame_i, frame, identify_result_queue))

                # 2 track 分发用于追踪的数据
                # resize
                frame = cv2.resize(frame, (IMG_FOR_GPU_W, IMG_FOR_GPU_H))
                # 向队列里放用于追踪的数据
                frame_data = [cameraID, frame_i, frame]
                put_data_to_queue(track_frame_data_queue, frame_data, "All frame_data_for_body_queue")

        distribute_time = time.time() - start
        if LOG_FLAG:
            log.logger.info("Distribute time: " + str(distribute_time))
        if distribute_time > 0.25:
            pass
        else:
            time.sleep(0.25-distribute_time)


def track_gpu_calculate(camera_dict, batch_data_queue):
    """
    track的GPU计算
    :param camera_dict: 摄像头信息和数据, 用到：ID、 track_gpu_calculate_result_queue
    :param batch_data_queue:
    :return: 无, 结果放入track_gpu_calculate_result_queue
    """

    # openpose模型
    model = 'cmu'  # 'cmu /mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    model_w, model_h = 656, 368  # Recommends : 432x368 or 656x368 or 1312x736'
    e = TfPoseEstimator(get_graph_path(model), target_size=(model_w, model_h))
    # deepsort_gpu模型
    deepsort_preprocess = DeepSortPreprocess()

    img_for_gpu_w, img_for_gpu_h = IMG_FOR_GPU_W, IMG_FOR_GPU_H
    img_original_w, img_original_h = IMG_ORIGINAL_W, IMG_ORIGINAL_H
    w_ratio, h_ratio = img_original_w / img_for_gpu_w, img_original_h / img_for_gpu_h

    def _gpu_calculate(img_for_gpu):
        # openpose检测人体框=============================================================================
        start_openpose = time.time()
        humans = e.inference(img_for_gpu, resize_to_default=(model_w > 0 and model_h > 0),
                             upsample_size=4.0)
        body_box_list = []
        other_data_list = []
        for human in humans:
            result = human.get_useful_data(img_for_gpu_w, img_for_gpu_h, img_original_w,
                                           img_original_h)
            if result:
                body_box = result['body_box']
                other_data = result['other_data']
                body_box_list.append(body_box)
                other_data_list.append(other_data)
        boxes = body_box_list
        # print("openpose人体框检测时间: ", time.time() - start_openpose)

        # deepsort gpu计算===============================================================================
        start_deepsort_gpu = time.time()
        features = deepsort_preprocess.get_features(img_for_gpu, boxes)
        # print("deepsort_gpu计算时间： ", time.time() - start_deepsort_gpu)

        # gpu计算的人体框resize到原始图像中==================================================================
        boxes = resize_boxes(boxes, w_ratio, h_ratio)
        return boxes, features, other_data_list

    img_init = np.zeros([img_for_gpu_w, img_for_gpu_h, 3], dtype=np.uint8)
    _gpu_calculate(img_init)
    print('模型创建成功')

    camera_id_list = list(camera_dict.keys())
    gpu_calculate_result_dict = {cameraID: camera_dict[cameraID].track_gpu_calculate_result_queue
                                 for cameraID in camera_id_list}

    while True:
        start = time.time()
        frame_data = get_data_from_queue(batch_data_queue, 0.3, "TrackGpuCalculate: frame_data_for_body_queue")
        # 如果没有读到数据
        if not frame_data:
            continue
        # 如果没有读到正确的数据
        if not (isinstance(frame_data, list) and len(frame_data) == 3 and frame_data[2] is not None):
            print("TrackGpuCalculate: ", "数据错误")
            continue

        cameraID, frame_i, frame = frame_data
        # 获得结果存放队列
        gpu_calculate_result_queue = gpu_calculate_result_dict[cameraID]
        # track_gpu 计算track_gpu并传入cpu计算接收队列
        boxes, features, other_data_list = _gpu_calculate(frame)
        put_data_to_queue(gpu_calculate_result_queue, [cameraID, frame_i, boxes, features, other_data_list], "TrackGpuCalculate: gpu_calculate_result_queue")
        if LOG_FLAG:
            log.logger.info("TrackGpuCalculate time:" + str(time.time() - start))


def track_cpu_calculate_thread(camera: CameraInfoAndData):
    """
    track的cpu计算

    :param camera:
    :return:
    """
    gpu_calculate_result_queue = camera.track_gpu_calculate_result_queue
    cpu_calculate_result_queue = camera.all_result_queue

    deepsort = DeepSort()
    while True:
        start = time.time()
        gpu_calculate_result = get_data_from_queue(gpu_calculate_result_queue, 0.4, "TrackCpuCalculate: gpu_calculate_result")
        # 如果没有读到数据
        if not gpu_calculate_result:
            continue
        cameraID, frame_i, boxes, features, other_data_list = gpu_calculate_result
        # deepsort_cpu
        track_new_id_list, track_delete_id_list, not_confirmed_detected_track, detected_track \
            = deepsort.update(boxes, features, other_data_list)
        # 往队列里放结果
        data = ['track_result', cameraID, frame_i, track_new_id_list, track_delete_id_list,
                not_confirmed_detected_track, detected_track]
        put_data_to_queue(cpu_calculate_result_queue, data, "TrackCpuCalculate: gpu_calculate_result_queue")
        if LOG_FLAG:
            log.logger.info("TrackCpuCalculate time:" + str(time.time() - start))


def track_cpu_calculate(camera_dict):
    """
    track的CPU计算,每个摄像头对应一个线程

    :param camera_dict:
    :return:
    """
    camera_id_list = list(camera_dict.keys())

    track_cpu_calculate_thread_list = []
    for camera_id in camera_id_list:
        track_cpu_calculate_thread_list.append(
            threading.Thread(target=track_cpu_calculate_thread, args=(camera_dict[camera_id],)))

    for thread in track_cpu_calculate_thread_list:
        thread.start()
    for thread in track_cpu_calculate_thread_list:
        thread.join()

def result_calculate_thread(camera:CameraInfoAndData, result_queue):
    """
    接收track和identify结果，进行person_manage、match和area_judge
    :param camera:
    :param result_queue:
    :return:
    """
    cameraID = camera.ID
    camera_size = camera.size
    area_info_list = camera.area_info_list
    track_and_identify_result_queue = camera.all_result_queue

    person_manage = PersonsManage(cameraID)
    match = Match(mode=2)
    area_judge = AreaJudge(camera_size, area_info_list, mode=2)

    def _deal_track_result(track_new_id_list, track_delete_id_list, not_confirmed_detected_track_list, detected_tracks_list):
        """
        处理跟踪结果,更新顾客信息(body_box, area_id)

        :param track_new_id_list: 新出现的track
        :param track_delete_id_list: 要删除的track
        :param not_confirmed_detected_track_list: 未确认的track
        :param detected_tracks_list: 确认了的track
        :return:
        """
        for track_id in track_new_id_list:  # 新出现的track
            person_manage.add_person_use_trackID(track_id)
        for track_id in track_delete_id_list:  # 要删除的track
            person_manage.delete_person_use_trackID(track_id)
        for track_data in not_confirmed_detected_track_list:  # 未确认的track
            # 解析track_data
            track_id = track_data['trackID']
            track_body_box = track_data['body_box']
            # person_manage更新body_box
            person_manage.update_person_body_box(track_id, track_body_box)
            # person_manage获取要输出的数据
            send_data_flag, data = person_manage.get_person_output_data(track_id)
            if send_data_flag:
                put_data_to_queue(result_queue, data, "ResultCalculate: result_queue")
        for track_data in detected_tracks_list:  # 确认了的track
            # 解析track_data
            track_id = track_data['trackID']
            track_body_box = track_data['body_box']
            track_other_data = track_data['other_data']
            # 判断区域
            body_data_for_area_judge = track_data
            area_id = area_judge.judge(body_data_for_area_judge)
            # person_manage更新body_box, areaID
            person_manage.set_person_areaID(track_id, area_id)
            person_manage.update_person_body_box(track_id, track_body_box)
            # person_manage获取要输出的数据
            send_data_flag, data = person_manage.get_person_output_data(track_id)
            if send_data_flag:
                put_data_to_queue(result_queue, data, "ResultCalculate: result_queue")

    def _deal_match_result(match_result):
        """
        处理匹配结果,更新顾客信息(face_id, age,gender)

        :param match_result:
        :return:
        """
        for trackID, face_data in match_result:
            faceID = face_data['faceID']
            age = face_data['age']
            gender = face_data['gender']
            if faceID:
                person_manage.set_person_faceID(trackID, faceID)
            if age:
                person_manage.set_person_age(trackID, age)
            if gender:
                person_manage.set_person_gender(trackID, gender)
    while True:
        start = time.time()
        result = get_data_from_queue(track_and_identify_result_queue, 0.4,
                                     "ResultCalculate: track_and_identify_result_queue")
        if not result:
            continue

        # 处理track信息
        if isinstance(result, list) and len(result) > 1 and result[0] == 'track_result' and len(result) == 7:
            _, cameraID, frame_i, track_new_id_list, track_delete_id_list, not_confirmed_detected_track_list, detected_tracks_list = result
            print('body_frame_i', frame_i)
            #
            _deal_track_result(track_new_id_list, track_delete_id_list, not_confirmed_detected_track_list, detected_tracks_list)
            # 匹配
            match_result = match.body_frame_update(frame_i, detected_tracks_list)
            _deal_match_result(match_result)
        # 处理identify信息
        elif isinstance(result, list) and len(result) > 1 and result[0] == 'identify_result' and len(result) == 4:
            _, cameraID, frame_i, face_data = result
            print('face_frame_i', frame_i)
            # face body 匹配
            match_result = match.face_frame_update(frame_i, face_data)
            _deal_match_result(match_result)
        # 错误信息
        else:
            print("ResultCalculate: get track_and_identify_result_queue, but data is wrong")
        if LOG_FLAG:
            log.logger.info("ResultCalculate time:" + str(time.time() - start))


def result_calculate(camera_dict, result_send_queue):
    """
    接收track和identify结果,进行person_manage、match和area_judge, 每个摄像头对应一个线程
    :param camera_dict:
    :param result_send_queue:
    :return:
    """
    camera_id_list = list(camera_dict.keys())

    track_cpu_calculate_thread_list = []
    for camera_id in camera_id_list:
        track_cpu_calculate_thread_list.append(
            threading.Thread(target=result_calculate_thread, args=(camera_dict[camera_id],result_send_queue)))

    for thread in track_cpu_calculate_thread_list:
        thread.start()
    for thread in track_cpu_calculate_thread_list:
        thread.join()


def result_send(areaID_list, result_queue):
    """
    发送结果到数据库
    :param result_queue:
    :return:
    """
    from docs.conf import MESSAGE_SEND_HOST, MESSAGE_SEND_PORT, MESSAGE_SEND_QNAME_FOR_TRACK, \
        MESSAGE_SEND_QNAME_FOR_AREA_OCCUPANCY
    msg_sender_for_track = MsgSender(host=MESSAGE_SEND_HOST, port=MESSAGE_SEND_PORT, qname=MESSAGE_SEND_QNAME_FOR_TRACK)
    msg_sender_for_occupancy = MsgSender(host=MESSAGE_SEND_HOST, port=MESSAGE_SEND_PORT,
                                         qname=MESSAGE_SEND_QNAME_FOR_AREA_OCCUPANCY)
    data_fusion = AreaDataFusion(areaID_list, history_time_used=7)
    while True:
        track_result = get_data_from_queue(result_queue, 2, "ResultSend: result_queue")
        if track_result:
            # 发送track信息
            # print(track_result)
            msg_sender_for_track.send_result(json.dumps(track_result))
            data_fusion.update(track_result)
        else:
            pass

        # 发送occupancy信息
        occupancy_result = data_fusion.get_data()
        for data in occupancy_result:
            # print(data)
            msg_sender_for_occupancy.send_result(json.dumps(data))


if __name__ == '__main__':
    # 设置模式（测试/部署），设置使用的摄像头
    from docs.conf import TEST_FLAG
    if TEST_FLAG:  # 从本地读参数和视频
        from docs.conf import CAMERAS_INFO

        CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}
    else:  # 从数据库读参数，网络摄像头读视频
        from persons_track.utils.modules.dbutil import MySQLPlugin
        from docs.conf import DB_CONF

        db = MySQLPlugin(host=DB_CONF['server'], user=DB_CONF['user'], password=DB_CONF['pwd'], db=DB_CONF["db"])
        CAMERAS_INFO = db.query_camera_data()  # 从数据库获取摄像头info
        CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}

    cameraID_list = [120, 121]  # 119, 114  # 109, 118  # 120, 121
    areaID_list = []
    CAMERA_NUM = len(cameraID_list)
    camera_dict = {}  # key:cameraID, value:camera相关信息和数据
    for cameraID in cameraID_list:
        camera_dict[cameraID] = CameraInfoAndData(CAMERAS_INFO[cameraID])
    for cameraID in camera_dict:
        area_info_list = camera_dict[cameraID].area_info_list
        for area_info in area_info_list:
            areaID_list.append(area_info[0])

    areaID_list = list(set(areaID_list))
    print('cameraID_list', cameraID_list)
    print('areaID_list', areaID_list)

    GPU_NUM = 1
    CAMERA_NUM_PER_GPU = math.ceil(CAMERA_NUM / GPU_NUM)

    # 所有进程队列
    ProcessList = []
    batch_data_queue = mp.Queue(5)
    result_send_queue = mp.Queue(20)

    # ======================================================进程创建=====================================================
    # 步骤1==============================================================================================================
    # 数据分发
    frame_data_get_and_distribute_process_process = \
        mp.Process(target=frame_data_get_and_distribute,
                   args=(camera_dict, batch_data_queue))
    ProcessList.append(frame_data_get_and_distribute_process_process)
    # 步骤2==============================================================================================================
    # 步骤2.1====================================
    # 用于track的gpu计算
    gpu_calculate_process = mp.Process(target=track_gpu_calculate, args=(camera_dict, batch_data_queue))
    ProcessList.append(gpu_calculate_process)
    # 步骤2.2======================================
    # 用于track的cpu计算
    cpu_calculate_process = mp.Process(target=track_cpu_calculate, args=(camera_dict,))
    ProcessList.append(cpu_calculate_process)
    # 步骤3==============================================================================================================
    # track和identify结果汇总
    result_calculate_process = mp.Process(target=result_calculate, args=(camera_dict, result_send_queue))
    ProcessList.append(result_calculate_process)
    # 步骤4==============================================================================================================
    # 发送结果
    result_send_process = mp.Process(target=result_send, args=(areaID_list, result_send_queue))
    ProcessList.append(result_send_process)

    # ======================================================进程start====================================================
    # 先创建gpu计算进程，初始化费时较长
    gpu_calculate_process.start()
    time.sleep(20)
    # 创建其他进程
    frame_data_get_and_distribute_process_process.start()
    cpu_calculate_process.start()
    result_calculate_process.start()
    result_send_process.start()

    # ======================================================进程join======================================================
    for process in ProcessList:
        process.join()