import copy
import threading
from typing import Any

import cv2
import numpy as np
import zmq
# from PySide6.QtCore import QTimer, Qt, QThread, Signal, Slot
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtWidgets import QWidget, QGraphicsScene, QApplication
# from ui.LeafBot_ui import Ui_LeafBotForm
# from module.AI_model import Yolo, MobileSAM
# from module.msg import ARMTASK, Msg
# from utils import ssh, image_process, leaf, file
# from module.kinect import Kinect
import requests
from _cffi_backend import buffer


def recv_array(socket, flags=0, copy=True, track=False):
    """Receive a numpy array over a ZeroMQ socket."""


    # Receive the actual data
    msg = socket.recv(flags=flags, copy=copy, track=track)

    # Convert received bytes to a NumPy array
    A = np.frombuffer(msg, dtype=np.uint8)

    # Reshape array to its original shape
    return A.reshape(480, 640, 3)


class Communicator:
    def __init__(self):
        self.context = zmq.Context()
        self.pair_cam = self.context.socket(zmq.SUB)
        self.pair_cam2 = self.context.socket(zmq.SUB)

        self.pair_cam.linger = 1
        self.pair_cam.setsockopt(zmq.SUBSCRIBE, b'')
        self.pair_cam.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self.pair_cam.setsockopt(zmq.RCVHWM, 1)

        self.pair_cam.connect("tcp://192.168.101.12:5555")

        self.pair_cam2.setsockopt(zmq.SUBSCRIBE, b'')
        self.pair_cam2.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self.pair_cam2.linger = 1
        self.pair_cam2.setsockopt(zmq.RCVHWM, 1)

        self.pair_cam2.connect("tcp://192.168.101.12:5556")
        self.pair_arm = self.context.socket(zmq.PAIR)
        self.pair_arm.bind("tcp://192.168.101.14:3333")

        car_ip_addr = '192.168.101.19'
        self.url = "http://" + car_ip_addr + "/js?json="

    def get_data(self):
        return recv_array(self.pair_cam)

    def get_data_depth(self):
        return self.pair_cam2.recv_pyobj()


class Data(threading.Thread):
    def __init__(self):
        super(Data, self).__init__()
        self.data = None
        self.com = Communicator()

    def run(self):
        while True:
            self.data = recv_array(self.com.pair_cam)


data_stream = Data()
data_stream.start()
while True:
    print(data_stream.data)
