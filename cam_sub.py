#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
sys.path.append('/home/hao/Desktop/wur_navcar/')
print(sys.path)
import time
import zmq
from module.camera import Camera
from utils.sensor import sensor_detect
import threading
import time


cap = Camera()
cap.run()
print('camera loaded')


class producer:
    def __init__(self):
        context = zmq.Context()
        sensor_detect()
        self.zmq_socket_rgb = context.socket(zmq.PUB)
        self.zmq_socket_rgb.setsockopt(zmq.SNDHWM, 1)  
        self.zmq_socket_rgb.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self.zmq_socket_rgb.linger = 1
        self.zmq_socket_d = context.socket(zmq.REP)
        self.zmq_socket_rgb.bind("tcp://192.168.101.12:5555")
        self.zmq_socket_d.bind("tcp://192.168.101.12:5556")

    # Start your result manager and workers before you start your producers
    def send_rgb(self):
        while True:
            rgb_img= cap.get_rgb_images()
            # self.zmq_socket_rgb.send_pyobj(rgb_img,copy=False, track=False)
            self.zmq_socket_rgb.send_pyobj(rgb_img,copy=False, track=False)
            
    def send_all(self):
        while True:
            self.zmq_socket_d.recv_pyobj()
            if self.zmq_socket_d.recv_pyobj()=='detect':
                sensor_detect()
                self.zmq_socket_d.send_pyobj(1)
            else:
                depth_img= cap.get_depth_images()
                self.zmq_socket_d.send_pyobj(depth_img)
            time.sleep(1)

cam = producer()
t1 = threading.Thread(target=cam.send_rgb)
t2 = threading.Thread(target=cam.send_all)

t2.start()
t1.start()
