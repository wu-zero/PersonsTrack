__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"

from rsmq import RedisSMQ


class MsgSender:
    def __init__(self, host, port='6379', qname='message_sender'):
        self.queue = RedisSMQ(host=host, port=port, qname=qname)
        self.msg = []
        try:
            # 删除queue如果已经存在
            self.queue.deleteQueue().exceptions(False).execute()
        except Exception as e:
            print(e)
        try:
            # 创建queue
            self.queue.createQueue(delay=0).vt(0).maxsize(-1).execute()
        except Exception as e:
            print(e)

    def send_result(self, result):
        message_id = self.queue.sendMessage(delay=0).message(str(result)).execute()
        self.msg.append(message_id)
        if len(self.msg) > 20:
            rt = self.queue.deleteMessage(id=self.msg[0]).execute()
            if rt:
                print("RedisSMQ send_result block")
            del self.msg[0]


if __name__ == '__main__':
    import time
    import json
    ply_sender = MsgSender(host="192.168.31.33", qname='track')
    result = {'cameraID': 1,
              'trackID': '2',
              'faceID': '3',
              'track_list': [{'timestamp': 1234567891234, 'rect': [23.0, 23, 23, 23]},
                             {'timestamp': 1234567891235, 'rect': [23, 23.0, 23, 23]}]
              }
    while True:
        time.sleep(1)
        ply_sender.send_result(json.dumps(result))
