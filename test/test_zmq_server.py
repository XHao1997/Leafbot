#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#
import copy
import time

import zmq
import threading
from utils import ssh


# class Communicator:
#     def __init__(self):
#         self.context = zmq.Context()
#         self.pair_arm = self.context.socket(zmq.SUB)
#         self.pair_arm.connect("tcp://192.168.101.11:6666")
#
#     def get_data_from_robot(self):
#         return self.pair_arm.recv()


if __name__ == '__main__':
    context = zmq.Context().instance()
    pair_arm = context.socket(zmq.SUB)
    pair_arm.setsockopt(zmq.SUBSCRIBE, b"")
    pair_arm.connect("tcp://192.168.101.12:7777")
    print('done')
    while True:
        msg = pair_arm.recv_pyobj()
        print(msg)

