__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"


if __name__ == "__main__":
    from persons_track.utils.modules.dbutil import MySQLPlugin

    from persons_track.CameraInfoAndData import CameraInfoAndData
    from docs.conf import DB_CONF

    db = MySQLPlugin(host=DB_CONF['server'], user=DB_CONF['user'], password=DB_CONF['pwd'], db=DB_CONF["db"])

    CAMERAS_INFO = db.query_camera_data()  # 从数据库获取摄像头info
    CAMERAS_INFO = {camera_info['cameraID']: camera_info for camera_info in CAMERAS_INFO}

    cameraID_list = [106, 107, 108, 105, 119, 109, 120, 114, 118, 121]  # 106,107, 108, 105
    CAMERA_NUM = len(cameraID_list)
    camera_dict = {}  # key:cameraID, value:camera相关信息和数据
    for cameraID in cameraID_list:
        camera_dict[cameraID] = CameraInfoAndData(CAMERAS_INFO[cameraID])

    for cameraID in cameraID_list:
        print(camera_dict[cameraID])
