# 发布者
import zmq
import time
from module.msg import Msg, ARMTASK

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

# while True:
for i in range(1000):
    msg = Msg()
    msg.task = ARMTASK.MOVE_ALL_JOINT
    news = msg

    socket.send_pyobj(news)
    time.sleep(1)