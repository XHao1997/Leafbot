from matplotlib import pyplot as plt
import zmq
from module.AI_model import Yolo, MobileSAM
import time
from utils import ssh, image_process
import os

PROJECT_PATH = os.getcwd()
print(PROJECT_PATH)
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Communicator(object):
    def __init__(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.consumer_receiver = self.context.socket(zmq.PULL)
        self.publisher.bind("tcp://192.168.101.14:5555")
        self.consumer_receiver.connect("tcp://192.168.101.12:5557")
        self.yolo = Yolo()
        self.sam = MobileSAM()
        self.launch()
        time.sleep(1)

    def rgb_image(self):
        image = self.consumer_receiver.recv_pyobj()
        return image

    def sam_mask(self):
        image = self.consumer_receiver.recv_pyobj()
        yolo_results = self.yolo.predict(image)
        mask = self.sam.predict(image, yolo_results)
        return mask

    def yolo_image(self):
        image = self.consumer_receiver.recv_pyobj()
        yolo_results = self.yolo.predict(image)
        yolo_img = image_process.draw_yolo_frame_cv(image, yolo_results)
        return yolo_img

    def send_cmd_to_arm(self):
        print("send_cmd_to_arm called")
        self.publisher.send_string("hello world")
        time.sleep(1)

    @staticmethod
    def launch():
        ssh.run_remote_stream()
        print('cam_server started')


if __name__ == '__main__':
    client = Communicator()
    # while True:
    #     cv2.imshow('frame',cv2.cvtColor(client.yolo_image(),cv2.COLOR_BGR2RGB))
    #     cv2.waitKey(1)
    mask = client.sam_mask()
    print(mask.shape)
    plt.imshow(mask)
    plt.show()
