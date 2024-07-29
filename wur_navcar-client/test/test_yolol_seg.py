#!/usr/bin/env python
import os
import sys
import cv2
import PIL 
import time
import matplotlib.pyplot as plt
import numpy as np
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
from utils import leaf, image_process
from module.AI_model import Yolo, MobileSAM
from module.kinect import Kinect 
kinect = Kinect()
# Directory where files are saved
directories = {
    'rgb': 'rgb_cali/',
    'depth': 'depth_cali',
    'eye_to_hand': 'eye_to_hand/',
    'joint1_nn': 'joint1_nn/'
}
depth_image = cv2.imread('depth_81.png',cv2.IMREAD_UNCHANGED)

rgb_img = cv2.imread('rgb_81.png').astype(np.uint8)
image_yolo = rgb_img
yolo = Yolo()
sam = MobileSAM()


yolo_result = yolo.predict(image_yolo)
print(yolo_result)
yolo.visualise_result(image_yolo, yolo_result)
# print(depth_image.shape)
start= time.time()
sam_mask = sam.predict(image_yolo,yolo_result)
end= time.time()
print("sam model takes {:.2f} seconds".format(end - start))
# cv2.imshow('sam_mask',sam_mask)
cv2.waitKey(0)
# plt.imshow(rgb_img)
# plt.show()
plt.imshow(sam_mask)
print(len(yolo_result))



for id in range(len(yolo_result)):
    result = yolo_result[id]
    chosen_leaf_roi = image_process.get_yolo_roi(sam_mask, result)
    plt.imshow(chosen_leaf_roi)
    contours = leaf.get_cnts(chosen_leaf_roi)
    mask, _ = leaf.get_incircle(chosen_leaf_roi, contours)
    picking_point = kinect.get_point_xyz(mask, rgb_img, depth_image)
    print(picking_point)



