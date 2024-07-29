#!/usr/bin/env python
import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

import time
import cv2
import numpy as np
from utils import vision, cali
import open3d as o3d


def prepocess_depth_img(depth_img):
    rows, cols = depth_img.shape
    M = np.float32([[1, 0, 4], [0, 1, 3]])
    depth_img = cv2.warpAffine(depth_img, M, (cols, rows))
    return depth_img


class Kinect:
    def __init__(self):
        # Defining the intrinsic and distortion matrices for the IR and RGB cameras,as 
        # well as other related parameters and matrices used in the camera calibration 
        # and image processing tasks.
        self.__ir_intrinsic_matrix = np.array(
            [[580.938217500424, 0, 317.617632886121],
             [0, 579.637279926675, 246.729727555759], [0, 0, 1]])
        self.__ir_distortion_matrix = np.array(
            [-0.203726937212412, 0.888177301091217, 0.002407122300079, -0.005097629434545, 0])
        self.__rgb_intrinsic_matrix = np.array(
            [[516.807042827898, 0, 334.465952581250],
             [0, 515.754575693376, 256.996105088827], [0, 0, 1]])
        self.__rgb_distortion_matrix = np.array(
            [0.205630360349696, -0.666613712966367, 0.007397174890620, -0.003644398316505, 0])

        self.__A = np.array([[
            0.999960280047999, -0.00552879643268073, -0.00699076078363541,
            -25.2137311361220
        ],
            [
                0.00555933604534934, 0.999975056126553,
                0.00435670832530621, 0.158410973817053
            ],
            [
                0.00696649905353596, -0.00439539926546948,
                0.999966073602617, -0.282858204409621
            ]])
        self.__new_rgb_intrinsic_matrix = None
        self.__new_ir_intrinsic_matrix = self.__ir_intrinsic_matrix
        self.__imgsz = [640, 480]

        return

    # ir to depth offset, reference: https://wiki.ros.org/kinect_calibration/technical

    def map_dist_to_rgb(self, dist_img):
        E = self.__A
        K = self.__new_rgb_intrinsic_matrix
        T_d2rgb = K.dot(E)
        X, Y, Z = self.pixel_to_world(dist_img)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        # Combine into a single array of 3D points
        points_3d = np.vstack((X_flat, Y_flat, Z_flat)).T
        # Remove rows where Z < 0
        points_3d = points_3d[points_3d[:, 2] >= 0]
        points_3d = np.append(points_3d.T, np.ones((1, points_3d.shape[0])), axis=0).T
        raw_pixel = T_d2rgb.dot(points_3d.T).T
        rgb_pixel = np.round((raw_pixel / (raw_pixel[:, -1]).reshape(-1, 1))).astype(int)
        return rgb_pixel

    def get_distance(self, depth_img):
        # dist = 0.1236 * np.tan((depth_img) / 2842.5 + 1.1863) * 1000-37
        dist = 0.075 * 580 / (1090 - depth_img) * 8 * 1000
        dist[dist < 500] = 0
        dist[dist > 1500] = 0
        return dist

    def pixel_to_world(self, dist_img):
        x = np.tile(np.arange(640), (480, 1))
        y = np.tile(np.arange(480).reshape(-1, 1), (1, 640))
        Z = dist_img[y, x]
        cx = self.__new_ir_intrinsic_matrix[0, 2]
        cy = self.__new_ir_intrinsic_matrix[1, 2]
        fx = self.__new_ir_intrinsic_matrix[0, 0]
        fy = self.__new_ir_intrinsic_matrix[1, 1]
        X = (x - cx) * (Z) / fx
        Y = (y - cy) * (Z) / fy
        return X, Y, Z

    def undistort(
            self, distorted_img, camera_type):
        # Correcting the distortion
        if camera_type == 'rgb':
            self.__new_rgb_intrinsic_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.__rgb_intrinsic_matrix, self.__rgb_distortion_matrix,
                self.__imgsz, 1, self.__imgsz)
            undistorted_img = cv2.undistort(
                distorted_img, self.__rgb_intrinsic_matrix,
                self.__rgb_distortion_matrix, None,
                self.__new_rgb_intrinsic_matrix)  # Correcting the distortion
        elif camera_type == 'ir':
            self.__new_ir_intrinsic_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.__ir_intrinsic_matrix, self.__ir_distortion_matrix,
                self.__imgsz, 1, self.__imgsz)
            undistorted_img = cv2.undistort(
                distorted_img, self.__ir_intrinsic_matrix,
                self.__ir_distortion_matrix, None,
                self.__new_ir_intrinsic_matrix)  # Correcting the distortion
        return undistorted_img

    def get_centroid_coordindate(self, pcd):
        # Assigning the points
        points_array = np.asarray(pcd.points)
        centroid_coordindate = np.median(points_array, axis=0)
        return centroid_coordindate

    def get_point_xyz(self, roi_mask, rgb_img, depth_img, mask_yolo_exp):
        # ir to depth offset, reference: https://wiki.ros.org/kinect_calibration/technical
        depth_img_post = prepocess_depth_img(depth_img)
        depth_img_post = cv2.bitwise_and(depth_img_post, depth_img_post, mask=mask_yolo_exp)

        roi_img = cv2.bitwise_and(rgb_img, rgb_img, mask=roi_mask)
        # undistort rgb and depth image to get new camera matrix
        roi_img = self.undistort(roi_img, 'rgb')
        _ = self.undistort(depth_img_post, 'ir')
        # convert disparity to distance
        dist_img = self.get_distance(depth_img_post)
        # from distance and ir_intrinsic calculate xyz in camera's world frame
        x, y, z = self.pixel_to_world(dist_img)
        points_3d = vision.create_point3d_from_xyz(x, y, z)
        rgb_pixel = self.map_dist_to_rgb(dist_img)
        leaf_pcd = vision.filter_roi_in_pcd(roi_img, rgb_pixel, points_3d)
        # o3d.visualization.draw_geometries([leaf_pcd])
        self.get_centroid_coordindate(leaf_pcd)
        xyz = (self.get_centroid_coordindate(leaf_pcd))
        return xyz
