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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def area_judge_and_show(cameraID, camera_address, camera_rotate, camera_size, area_info_list):
    import cv2
    import time

    from third_party.tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
    from third_party.tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

    from persons_track.utils.camera_capture import VideoCapture
    from persons_track.utils.others import resize_box, box_tlwh_to_tlbr
    from persons_track.AreaJudge import AreaJudge

    video_capture = VideoCapture(cameraID, camera_address, camera_rotate)
    model = 'cmu'  # 'cmu /mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    resolution = '656x368'  # Recommends : 432x368 or 656x368 or 1312x736'
    model_w, model_h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(model_w, model_h))
    area_judge = AreaJudge(img_shape=camera_size, area_info_list=area_info_list, mode=2)

    # 窗口设置
    cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    points = []
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    cv2.setMouseCallback('test', mouse)

    while True:
        start = time.time()
        cameraID, frame_i, frame = video_capture.read()
        if frame is None:
            print("get frame timeout")
            continue
        print("Frame_i", frame_i)
        print("读图片时间: ", time.time() - start)

        frame_original_h, frame_original_w = frame.shape[0:2]
        frame_for_gpu = cv2.resize(frame, (model_w, model_h))
        frame_for_gpu_h, frame_for_gpu_w = frame_for_gpu.shape[0:2]
        w_ratio, h_ratio = frame_original_w / frame_for_gpu_w, frame_original_h / frame_for_gpu_h
        # 检测人体框======================================================================================================
        start = time.time()
        humans = e.inference(frame_for_gpu, resize_to_default=(model_w > 0 and model_h > 0), upsample_size=4.0)
        # frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        body_data_for_area_judge_list = []
        for human in humans:
            result = human.get_useful_data(frame_for_gpu_w, frame_for_gpu_h, frame_original_w, frame_original_h)
            if result:
                body_box = result['body_box']
                other_data = result['other_data']

                body_box = box_tlwh_to_tlbr(body_box)
                body_box = resize_box(body_box, w_ratio, h_ratio)

                body_data_for_area_judg = {'body_box': body_box, 'other_data': other_data}

                body_data_for_area_judge_list.append(body_data_for_area_judg)
        print("openpose人体框检测时间: ", time.time() - start)

        # print(body_data_for_area_judge_list)
        # ============================================================================================
        # 结果显示
        area_judge.draw(frame)
        for body_data in body_data_for_area_judge_list:
            body_box = body_data['body_box']
            other_data = body_data['other_data']
            cv2.rectangle(frame, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (0, 255, 0), 2)

            for part in ['nose_point', 'rhip_point', 'lhip_point', 'rshoulder_point', 'lshoulder_point']:
                if other_data[part] is not None:
                    cv2.circle(frame, tuple(other_data[part]), 5, (255, 0, 0), thickness=-1)

            # 判断区域
            area = area_judge.judge(body_data)
            cv2.putText(frame, str(area), (body_box[0], body_box[1]), 0, 5e-3 * 200, (0, 255, 0), 2)

        # ===========================================================================================
        # 显示鼠标标记结果， 用于标注
        for point in points:
            x, y = point
            xy = "%d,%d" % (x, y)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(frame, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0), thickness=2)
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

    cameraID = 119

    # camera相关信息和q数据
    camera_info_and_data = CameraInfoAndData(CAMERAS_INFO[cameraID])
    camera_address = camera_info_and_data.url
    camera_frame_rate = camera_info_and_data.frame_rate
    camera_rotate = camera_info_and_data.rotate
    camera_size = camera_info_and_data.size
    area_info_list = camera_info_and_data.area_info_list
    frame_data_queue = camera_info_and_data.frame_data_queue  # 图像帧队列

    processes = [mp.Process(target=area_judge_and_show, args=(cameraID, camera_address, camera_rotate, camera_size, area_info_list))]

    [process.start() for process in processes]
    [process.join() for process in processes]






