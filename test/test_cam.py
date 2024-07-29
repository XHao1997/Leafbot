import numpy as np
import cv2 
import zmq
import threading
import multiprocessing
import time
cap = cv2.VideoCapture(0)


class producer:
    def __init__(self):
        context = zmq.Context()
        self.zmq_socket_rgb = context.socket(zmq.PUB)
        self.zmq_socket_rgb.setsockopt(zmq.SNDHWM, 1)  
        self.zmq_socket_rgb.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self.zmq_socket_rgb.linger = 1
        self.zmq_socket_rgb.bind("tcp://192.168.101.11:7777")

    def send_rgb(self):
        _,rgb_img = cap.read()
        rgb_img = cv2.resize(rgb_img, (32, 32))
        print(rgb_img)
        self.zmq_socket_rgb.send_pyobj(rgb_img,copy=True, track=False)
        time.sleep(0.1)

cam = producer()
while True:
    cam.send_rgb()