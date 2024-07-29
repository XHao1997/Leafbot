import time
import zmq
import random
import numpy as np
import cv2
from utils import ssh,image_process


def rgb2bgr(data):
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return data  # encode Numpay to Bytes string


def consumer():
    consumer_id = random.randrange(1, 10005)
    print
    "I am consumer #%s" % (consumer_id)
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://192.168.101.12:5557")
    # send work
    cv2.namedWindow('test', cv2.WINDOW_KEEPRATIO)
    while True:
        image = consumer_receiver.recv_pyobj()

        cv2.imshow('test', image_process.rgb2bgr(image))
        # if data % 2 == 0:
        #     consumer_sender.send_json(result)
        cv2.waitKey(1)


ssh.run_remote_stream()
consumer()
