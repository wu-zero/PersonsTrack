__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import dlib
import cv2

detector = dlib.get_frontal_face_detector() #获取人脸分类器


def detect_face(img):
    # 摘自官方文档：
    # image is a numpy ndarray containing either an 8bit grayscale or RGB image.
    # opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
    b, g, r = cv2.split(img)    # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片

    dets = detector(img2, 1) #使用detector进行人脸检测 dets为返回的结果
    result = []
    for index, face in enumerate(dets):
        result.append([face.left(), face.top(), face.right(), face.bottom()])
    return result


if __name__ == '__main__':
    import os
    from docs.conf import DATA_PATH
    img = cv2.imread(os.path.join(DATA_PATH, '1.jpg'))
    faces = detect_face(img)
    for face in faces:
        cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (255, 255, 255), 3)
    cv2.imshow('test', img)
    cv2.waitKey()

