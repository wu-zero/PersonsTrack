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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def person_manage_and_show(cameraID, camera_address, camera_rotate, camera_size, area_info_list, areaID_list):
    import cv2
    import time

    from third_party.tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
    from third_party.tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

    from third_party.deep_sort_yolov3.deep_sort_model import DeepSortPreprocess, DeepSort

    from persons_track.utils.others import resize_boxes
    from persons_track.utils.camera_capture import VideoCapture
    from persons_track.AreaJudge import AreaJudge
    from persons_track.PersonManage import PersonsManage
    from persons_track.DataFusion import AreaDataFusion

    video_capture = VideoCapture(cameraID, camera_address, camera_rotate)
    model = 'cmu'  # 'cmu /mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    resolution = '656x368'  # Recommends : 432x368 or 656x368 or 1312x736'
    model_w, model_h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(model_w, model_h))

    deepsort_preprocess = DeepSortPreprocess()
    deepsort = DeepSort()
    area_judge = AreaJudge(img_shape=camera_size, area_info_list=area_info_list, mode=2)
    person_manage = PersonsManage(cameraID)
    area_data_fusion = AreaDataFusion(areaID_list)

    # 窗口设置
    cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 10.0, (2560, 1440))

    while True:
        start = time.time()
        cameraID, frame_i, frame = video_capture.read()
        if frame is None:
            print("get frame timeout")
            continue
        print("Frame_i", frame_i)
        print("读图片时间: ", time.time() - start)

        frame_for_gpu = cv2.resize(frame, (model_w, model_h))
        frame_original_h, frame_original_w = frame.shape[0:2]
        frame_for_gpu_h, frame_for_gpu_w = frame_for_gpu.shape[0:2]
        w_ratio, h_ratio = frame_original_w / frame_for_gpu_w, frame_original_h / frame_for_gpu_h

        # openpose检测人体框===============================================================================================
        start = time.time()
        humans = e.inference(frame_for_gpu, resize_to_default=(model_w > 0 and model_h > 0), upsample_size=4.0)
        # frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
        body_box_list = []
        other_data_list = []
        for human in humans:
            result = human.get_useful_data(frame_for_gpu_w, frame_for_gpu_h, frame_original_w, frame_original_h)
            if result:
                body_box = result['body_box']
                other_data = result['other_data']
                body_box_list.append(body_box)
                other_data_list.append(other_data)
        boxes = body_box_list
        print("opepose人体框检测时间: ", time.time() - start)

        # deepsort gpu计算===============================================================================================
        start = time.time()
        features = deepsort_preprocess.get_features(frame_for_gpu, boxes)
        print("deepsort_gpu计算时间： ", time.time() - start)

        # gpu计算的人体框resize到原始图像中=================================================================================
        boxes = resize_boxes(boxes, w_ratio, h_ratio)

        # deepsort cpu计算===============================================================================================
        start = time.time()
        track_new_id_list, track_delete_id_list, not_confirmed_detected_track_list, detected_track_list = deepsort.update(boxes, features, other_data_list)
        print("deepsort_cpu计算时间： ", time.time() - start)

        # 处理track数据===================================================================================================
        area_judge.draw(frame)
        for track_id in track_new_id_list:  # 新出现的track
            # person_manage添加人
            person_manage.add_person_use_trackID(track_id)
        for track_id in track_delete_id_list:  # 要删除的track
            # person_manage删除人
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
                area_data_fusion.update(data)
                print(data)
                for area_data in area_data_fusion.get_data():
                    print(area_data)

            # body_box 可视化
            cv2.putText(frame, str(person_manage.get_person_still_status_flag(track_id)), (int(track_body_box[0])+40, int(track_body_box[1])), 0, 5e-3 * 200,
                        (255, 255, 255), 2)

            cv2.rectangle(frame, (int(track_body_box[0]), int(track_body_box[1])),
                          (int(track_body_box[2]), int(track_body_box[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track_id), (int(track_body_box[0]), int(track_body_box[1])-10), 0, 5e-3 * 200,
                        (255, 255, 255), 2)

        for track_data in detected_track_list:  # 确认了的track
            # 解析track_data
            track_id = track_data['trackID']
            track_body_box = track_data['body_box']
            track_other_data = track_data['other_data']
            # 判断区域
            body_data_for_area_judge = {'body_box': track_body_box, 'other_data': track_other_data}
            area_id = area_judge.judge(body_data_for_area_judge)
            # person_manage更新body_box, areaID
            person_manage.set_person_areaID(track_id, area_id)
            person_manage.update_person_body_box(track_id, track_body_box)
            # person_manage获取要输出的数据
            send_data_flag, data = person_manage.get_person_output_data(track_id)
            if send_data_flag:
                area_data_fusion.update(data)
                print(data)
                for area_data in area_data_fusion.get_data():
                    print(area_data)

            # body_box 可视化
            for part in ['nose_point', 'rhip_point', 'lhip_point', 'rshoulder_point', 'lshoulder_point']:
                if track_other_data[part] is not None:
                    cv2.circle(frame, tuple(track_other_data[part]), 8, (255, 0, 0), thickness=-1)

            cv2.rectangle(frame, (int(track_body_box[0]), int(track_body_box[1])),
                          (int(track_body_box[2]), int(track_body_box[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (int(track_body_box[0]), int(track_body_box[1])-10), 0, 5e-3 * 200,
                        (0, 255, 0), 2)
            cv2.putText(frame, str(area_id), (int(track_body_box[0]), int(track_body_box[1])-50), 0, 5e-3 * 200,
                        (0, 255, 0), 2)
            cv2.putText(frame, str(person_manage.get_person_still_status_flag(track_id)),
                        (int(track_body_box[0]) + 100, int(track_body_box[1]) - 50), 0, 5e-3 * 200,
                        (0, 255, 0), 2)
        out.write(frame)

        cv2.imshow('test', frame)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
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

    cameraID = 120

    # camera相关信息和数据
    camera_info_and_data = CameraInfoAndData(CAMERAS_INFO[cameraID])
    camera_address = camera_info_and_data.url
    camera_frame_rate = camera_info_and_data.frame_rate
    camera_rotate = camera_info_and_data.rotate
    camera_size = camera_info_and_data.size
    area_info_list = camera_info_and_data.area_info_list
    frame_data_queue = camera_info_and_data.frame_data_queue  # 图像帧队列

    # area相关信息和数据
    areaID_list = []
    for area_info in area_info_list:
        areaID_list.append(area_info[0])
    areaID_list = list(set(areaID_list))

    # 开始process
    processes = [mp.Process(target=person_manage_and_show, args=(cameraID, camera_address, camera_rotate, camera_size, area_info_list, areaID_list))  # 追踪进程
                 ]

    [process.start() for process in processes]
    [process.join() for process in processes]






