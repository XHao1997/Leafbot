import os
import sys
import cv2
import numpy as np
import open3d as o3d

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)


def filter_roi_in_pcd(rgb_img, rgb_pixel, points_3d):
    roi_index = []
    points_3d_color = []
    for i, _ in enumerate(points_3d):
        u, v = rgb_pixel[i][:2]
        if u >= 640 or v >= 480:
            pass
        elif u < 0 or v < 0:
            pass
        else:
            pc = rgb_img[v, u]
            if not np.array_equal(pc, np.array([0, 0, 0])):
                roi_index.append(i)
                points_3d_color.append(pc)
    points_3d = np.asarray(points_3d)
    roi_3d = points_3d[roi_index]
    pcd = o3d.geometry.PointCloud()
    if len(points_3d_color) != 0:
        points_3d_color_filtered = np.asarray(points_3d_color)
        # Assigning the colors
        pcd.colors = o3d.utility.Vector3dVector(points_3d_color_filtered[:, [2, 1, 0]] / 255.0)
        pcd.points = o3d.utility.Vector3dVector(roi_3d[:, :3])
    # Visualizing the point cloud
        if pcd.points is not None:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=10)
    #         # if not ind:
    #         #     pcd = cl.select_by_index(ind)
    #         #     cl_final, ind_final = pcd.remove_statistical_outlier(nb_neighbors=20,
    #         #                                                          std_ratio=10)
            if not ind:
                pcd = cl.select_by_index(ind)
    return pcd


def create_point3d_from_xyz(x, y, z):
    # Flatten the matrices
    X_flat = x.flatten()
    Y_flat = y.flatten()
    Z_flat = z.flatten()
    # Combine into a single array of 3D points
    points_3d = np.vstack((X_flat, Y_flat, Z_flat)).T
    # Remove rows where Z < 0
    points_3d = points_3d[points_3d[:, 2] >= 0]

    return points_3d


def get_block_center(img, debug=False):
    """
    Detects the center of the largest connected component in the provided image.

    Args:
        img (numpy.ndarray): The input image in BGR color space.

    Returns:
        numpy.ndarray: Image with only the largest connected component.
    """
    # Convert image from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert image from RGB to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for HSV thresholding
    lower_hsv = np.array([0, 220, 102])
    upper_hsv = np.array([179, 255, 255])
    # Threshold the image to get the mask of the largest connected component
    mask_hsv = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    contours, hierarchy = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (largest connected component)
    max_contour = max(contours, key=cv2.contourArea)
    # Create an empty image to draw the ellipse
    ellipse_img = np.zeros_like(mask_hsv)
    # Fit an ellipse to the largest contour and get its center
    M = cv2.moments(max_contour)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # Draw a filled circle at the center of the ellipse
    cv2.circle(ellipse_img, center,
               5, 255, thickness=cv2.FILLED)
    # Apply the mask to the original image
    img_result_hsv = cv2.bitwise_and(img, img, mask=(ellipse_img).astype(np.uint8))
    if debug == True:
        img_result_hsv = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(ellipse_img).astype(np.uint8))
    # Create a named window
    if debug:
        cv2.namedWindow("mask")
        cv2.imshow('mask', img_result_hsv)
        cv2.waitKey(0)
    return img_result_hsv
