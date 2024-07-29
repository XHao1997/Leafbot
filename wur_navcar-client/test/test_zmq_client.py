import time
import zmq
import random
import numpy as np
import cv2
from utils import ssh, image_process


class LocalClient(object):
    def __init__(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://192.168.101.19:5555")
        self.luanch()
    def luanch(self):
        ssh.run_remote_stream()
        print('cam_server started')
        # receive work
        # while True:
        #     image = self.consumer_receiver.recv_pyobj()
        #     cv2.imshow('test', image_process.rgb2bgr(image))
        #     cv2.waitKey(1)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    def send_cmd_to_arm(self):
        print("send_cmd_to_arm called")
        self.publisher.send_string("hello world")
        time.sleep(1)

    def send_cmd_to_car(self, cmd):
        pass

    def send_cmd_to_cam(self, cmd):
        pass

    def cal_leaf_center(self):
        pass

    def cal_leaf_size(self):
        pass

#
client_server = LocalClient()
client_server.publisher.send_json({"T":0.1,"L":0.5,"R":0.5})
