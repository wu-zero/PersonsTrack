__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import time
import cv2
import base64

from persons_track.utils.modules.baidutil import BAIDUFace, calc_quality
from persons_track.utils.modules.repoutil import repository_process


FACE_IDENTITY_SCORE = 65
REPOSITORY_PROCESS_FLAG = True

baidu = BAIDUFace()
#===================================百度接口封装=========================
# 使用百度接口进行人脸检测
def baidu_face_detect(img):
    # 图片编码
    img_str = cv2.imencode('.jpg', img)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    img_b64 = str(base64.b64encode(img_str), 'utf-8')  # 编码成base64
    # 调用百度接口
    baidu_result = baidu.face_detect(img=img_b64,
                                     appid='default',
                                     image_type='BASE64',
                                     face_field='feature,age,gender,quality,angel,eye_status',
                                     max_face_num=10,
                                     face_type='LIVE',
                                     liveness_control='NONE')
    result = []
    if baidu_result == -1:
        return result
    for face in baidu_result['face_list']:
        angle = face['angle']
        eye_status = face['eye_status']
        face_quality = calc_quality(face)
        face_token = face['face_token']
        face_box = [int(face['location']['left']),
                    int(face['location']['top']),
                    int(face['location']['left'] + face['location']['width']),
                    int(face['location']['top'] + face['location']['height'])]
        age = face['age']
        gender = face['gender']['type']
        face_img = img[face_box[1]:face_box[3],face_box[0]:face_box[2]]
        result_i = {
            'angle': angle,
            'eye_status': eye_status,
            'face_quality': face_quality,
            'face_token': face_token,
            'face_box': face_box,
            'age': age,
            'gender': gender,
            'face_img': face_img
        }

        result.append(result_i)
    return result

# 使用百度接口进行人脸识别_face_token
def baidu_face_identity_use_face_token(face_token):
    # # 调用百度接口
    baidu_result = baidu.face_identity(img=face_token,
                                       image_type='FACE_TOKEN',
                                       group_id_list='member1,stranger')
    # 处理结果
    result = {}
    if baidu_result == -1:
        return result

    result['user_name'] = baidu_result['user_list'][0]['user_info']
    result['face_score'] = baidu_result['user_list'][0]['score']
    result['user_id'] = baidu_result['user_list'][0]['user_id']
    result['group_id'] = baidu_result['user_list'][0]['group_id']

    return result

# 使用百度接口进行人脸识别_image
def baidu_face_identity_use_img(img):
    # 图片编码
    img_str = cv2.imencode('.jpg', img)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    img_b64 = str(base64.b64encode(img_str), 'utf-8')  # 编码成base64
    # 调用百度接口
    baidu_result = baidu.face_identity(img=img_b64,
                                       image_type='BASE64',
                                       group_id_list='member1,stranger')
    # 处理结果
    result = {}
    if baidu_result == -1:
        return result

    result['user_name'] = baidu_result['user_list'][0]['user_info']
    result['face_score'] = baidu_result['user_list'][0]['score']
    result['user_id'] = baidu_result['user_list'][0]['user_id']
    result['group_id'] = baidu_result['user_list'][0]['group_id']

    return result

# ==========================================================================================
# 人脸识别
def face_identity(detect_result):
    img = detect_result['face_img']
    face_token = detect_result['face_token']

    identity_result = baidu_face_identity_use_face_token(face_token)
    # {'user_name': None, 'face_score': 33.273086547852, 'user_id': '8c6c4390e73a11e99281e0d55eee5246','group_id': 'member1'}
    if not identity_result:
        return None

    face_id = identity_result['user_id']
    face_score = identity_result['face_score']
    if face_score < FACE_IDENTITY_SCORE:
        return None
    else:
        return face_id

# 人脸识别并更新人脸库
def face_identity_with_repository_process(detect_result):
    img = detect_result['face_img']
    face_token = detect_result['face_token']

    identity_result = baidu_face_identity_use_face_token(face_token)
    # {'user_name': None, 'face_score': 33.273086547852, 'user_id': '8c6c4390e73a11e99281e0d55eee5246','group_id': 'member1'}
    if not identity_result:
        return None
    # 更新库，库中加新人
    repo_result = repository_process(img, detect_result, identity_result)
    # {'user_name': 'unknown_c16bd856289411eab067ac1f6b99012a', 'face_score': 100, 'user_id': 'c16bd856289411eab067ac1f6b99012a', 'group_id': 'stranger'}

    face_id = identity_result['user_id']
    face_score = identity_result['face_score']
    if face_score < FACE_IDENTITY_SCORE:
        return None
    else:
        return face_id

#
def face_identity_for_one_frame(detect_result_list):
    faceID_list = []
    face_num = len(detect_result_list)
    if face_num == 0:
        return faceID_list

    for detect_result in detect_result_list:
        if REPOSITORY_PROCESS_FLAG:
            result = face_identity_with_repository_process(detect_result)
        else:
            result = face_identity(detect_result)
        faceID_list.append(result)

    return faceID_list


def face_detect_and_identify(cameraID, frame_i, frame, recognition_result_queue):
    # print(cameraID, frame_i)
    face_detect_data_list = baidu_face_detect(frame)
    face_id_list = face_identity_for_one_frame(face_detect_data_list)
    # print(face_id_list)
    result = []
    face_num = len(face_detect_data_list)
    for i in range(face_num):
        face_id = face_id_list[i]
        face_box = face_detect_data_list[i]['face_box']
        age = face_detect_data_list[i]['age']
        gender = face_detect_data_list[i]['gender']
        result_i = {'faceID': face_id,
                    'age': age,
                    'gender': gender,
                    'face_box': face_box}
        result.append(result_i)
    if len(result) > 0:
        # print(result)
        recognition_result_queue.put(['identify_result', cameraID, frame_i, result])


if __name__ == '__main__':
    import os
    import multiprocessing as mp
    from docs.conf import DATA_PATH
    img = cv2.imread(os.path.join(DATA_PATH, '4.jpg'))
    LAST_ADD_TIME = mp.Manager().Value('d', 0)
    LAST_ADD_TIME_LOCK = mp.Manager().Lock()
    # 百度接口测试
    # print(baidu_face_detect(img))
    # print(baidu_face_identity_use_img(img))
    # print(baidu_face_identity_use_face_token('a262b902e2f9a0b2e6b41a6b7d76aec5'))
    # #
    detect_result = baidu_face_detect(img)
    print(detect_result)
    print(face_identity_for_one_frame(detect_result))
    import multiprocessing as mp
    face_detect_and_identify('1', '1', img, mp.Queue(1))




