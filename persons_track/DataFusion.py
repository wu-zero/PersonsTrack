__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

import time

SEND_TIME_INTERVAL = 3


class AreaData:
    def __init__(self, area_id, history_time_used=8):
        self._areaID = area_id
        self._history_time_used = history_time_used
        self._person_dict = {}
        self._history_timestamp = 0
        self._trackID_list_history = []
        self._faceID_list_history = []

    def _process_data_use_face_id_deduplicate(self):
        face_id_track_id_dict = {}
        delete_track_id_list = []
        for trackID in self._person_dict.keys():
            faceID = self._person_dict[trackID]['faceID']
            timestamp = self._person_dict[trackID]['timestamp']

            if faceID is None:
                continue

            if faceID not in face_id_track_id_dict.keys():
                face_id_track_id_dict[faceID] = [trackID, timestamp]
            else:
                trackId_old, timestamp_old = face_id_track_id_dict[faceID]
                if timestamp > timestamp_old:
                    delete_track_id_list.append(trackId_old)
                    face_id_track_id_dict[faceID] = [trackID, timestamp]
                else:
                    delete_track_id_list.append(trackID)
        for trackID in delete_track_id_list:
            self._person_dict.pop(trackID)

    def _process_data_use_current_timestamp_diff(self):
        time_now = time.time()
        delete_track_id_list = []
        for trackID in self._person_dict.keys():
            timestamp = self._person_dict[trackID]['timestamp']
            if time_now - timestamp > self._history_time_used:
                delete_track_id_list.append(trackID)
            else:
                pass

        for trackID in delete_track_id_list:
            self._person_dict.pop(trackID)

    def _data_change_flag(self, track_id_list, face_id_list):
        if time.time() - self._history_timestamp > SEND_TIME_INTERVAL:
            return True
        if len(track_id_list) != len(self._trackID_list_history) or len(face_id_list) != len(self._faceID_list_history):
            return True
        else:
            for track_id in track_id_list:
                if track_id not in self._trackID_list_history:
                    return True
            for face_id in face_id_list:
                if face_id not in self._faceID_list_history:
                    return True
            return False

    def update(self, track_id, face_id):
        time_now = time.time()
        if track_id in self._person_dict:
            self._person_dict[track_id]['timestamp'] = time_now
            if face_id is not None:
                self._person_dict[track_id]['faceID'] = face_id
        else:
            self._person_dict[track_id] = {'faceID': face_id, 'timestamp': time_now}

        self._process_data_use_face_id_deduplicate()

    def get_data(self):
        self._process_data_use_current_timestamp_diff()
        track_id_list = []
        face_id_list = []
        for track_id in self._person_dict.keys():
            face_id = self._person_dict[track_id]['faceID']
            track_id_list.append(track_id)
            if face_id is not None:
                face_id_list.append(face_id)

        change_flag = self._data_change_flag(track_id_list, face_id_list)

        if change_flag:
            self._history_timestamp = time.time()
            self._trackID_list_history = track_id_list
            self._faceID_list_history = face_id_list
            return track_id_list, face_id_list
        else:
            return None


class AreaDataFusion:
    def __init__(self, area_id_list, history_time_used=8):
        self._area_data_dict = {area_id: AreaData(area_id=area_id,history_time_used=history_time_used) for area_id in area_id_list}

    def update(self, data: dict):
        area_id = data.get('areaID', None)
        track_id = data.get('trackID', None)
        face_id = data.get('faceID', None)

        if area_id in self._area_data_dict:
            area_data = self._area_data_dict[area_id]
            area_data.update(track_id, face_id)

    def get_data(self):
        result = []
        for area_id in self._area_data_dict:
            data = self._area_data_dict[area_id].get_data()
            if data is not None:
                track_id_list, face_id_list = data
                data = {'areaID': area_id, 'timestamp': round(time.time()*1000), 'trackID_list': track_id_list, 'faceID_list': face_id_list}
                result.append(data)
        return result


if __name__ == '__main__':
    # # test AreaDate=============================================================
    # area_data = AreaData('1')
    # area_data.update('1', None)
    # time.sleep(1)
    # print(area_data.get_data())
    # area_data.update('2', '1')
    # time.sleep(1)
    # print(area_data.get_data())
    # time.sleep(1)
    # area_data.update('3', '2')
    # time.sleep(1)
    # print(area_data.get_data())
    # area_data.update('4', '2')
    # time.sleep(1)
    # print(area_data.get_data())
    # for i in range(6):
    #     time.sleep(1)
    #     print(area_data.get_data())


    # test AreaDateMerge=======================================================
    area_data_fusion = AreaDataFusion([1, 2])

    area_data_fusion.update({'areaID': 1, 'trackID': '1', 'faceID': None})
    time.sleep(1)
    area_data_fusion.update({'areaID': 2, 'trackID': '2', 'faceID': '1'})
    time.sleep(1)
    area_data_fusion.update({'areaID': 2, 'trackID': '3', 'faceID': '2'})
    time.sleep(1)
    area_data_fusion.update({'areaID': 3, 'trackID': '4', 'faceID': '3'})

    for i in range(6):
        print(i, '='*20)
        time.sleep(1)
        result = area_data_fusion.get_data()
        for data in result:
            print(data)
