import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def rgb2bgr(img):
    # Convert RGB image to BGR
    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    return bgr_image


# Convert DetectorResult objects to xyxy format
def convert_to_xyxy(result):
    x1 = result.x
    y1 = result.y
    x2 = result.x + result.width
    y2 = result.y + result.height
    return x1, y1, x2, y2


def get_yolo_roi(image, yolo_result):
    x0, y0, x1, y1 = convert_to_xyxy(yolo_result)
    x0 = x0 - 2
    y0 = y0 - 2
    x1 = x1 + 2
    y1 = y1 + 2
    mask = bbox2mask(x0, x1, y0, y1)
    # Apply the mask to the grayscale image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def convert_bbox_pct(bbox):
    x0, y0, x1, y1 = bbox
    src_pts = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    return src_pts.reshape(-1, 1, 2)


def get_xyxy_from_pct(pct):
    pct = pct.reshape(-1, 2)
    x0 = np.min(pct[:, 0])-5
    x1 = np.max(pct[:, 0])-5
    y0 = np.min(pct[:, 1])+5
    y1 = np.max(pct[:, 1])+5
    return x0, y0, x1, y1


def bbox2mask(x0, x1, y0, y1):
    mask = np.zeros((480, 640)).astype(np.uint8)
    # Create a mask for the YOLO bounding box
    mask = cv2.rectangle(mask, (x0, y0,), (x1, y1), (255, 255, 255), -1)
    return mask


def draw_yolo_frame_cv(image, yolo_results):
    results = yolo_results
    for i, result in enumerate(results):
        x0, y0, x1, y1 = convert_to_xyxy(result)
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(image, start_point, end_point, color=(255, 0, 0), thickness=2)
        cv2.putText(
            image,
            result.name + str(i),
            (int(x0), int(y0) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 30, 255),
            thickness=2
        )

    return image


# Define a function to draw rectangles on the image
def draw_yolo_frame_plt(ax, result, leaf_index, color='r'):
    rect = patches.Rectangle((result.x, result.y), result.width, result.height, linewidth=2, edgecolor=color,
                             facecolor='none')
    ax.add_patch(rect)
    ax.text(result.x, result.y - 5, result.name + str(leaf_index), fontsize=10, color=color)
    return


def remove_small_cnt(masks_final):
    contours, hierarchy = cv2.findContours(masks_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    bigger = max(contours, key=lambda item: cv2.contourArea(item))
    # Filter small contours
    contours_final = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > cv2.contourArea(bigger) / 10:
            contours_final.append(contours[i])
    return contours_final


def show_curent_pcd(rgb_img, depth_img):
    dist = disparity2dist(depth_img)
    pcd = None
    pass


# ir to depth offset, reference: https://wiki.ros.org/kinect_calibration/technical
def prepocess_depth_img(depth_img):
    rows, cols = depth_img.shape
    M = np.float32([[1, 0, 4], [0, 1, 3]])
    depth_img = cv2.warpAffine(depth_img, M, (cols, rows))
    return depth_img


def disparity2dist(disparity):
    """
    The function `disparity2dist` calculates the distance from a disparity value using a specific
    formula and applies certain threshold conditions.
    
    Args:
        disparity: Disparity refers to the difference in horizontal position of an object as seen by the
    left and right cameras in a stereo vision system. It is typically represented as a grayscale image
    where brighter pixels indicate objects that are closer to the cameras.
    
    Returns:
        The function `disparity2dist` returns the calculated distance values based on the input disparity
    values. The calculated distances are then filtered to be within the range of 500 to 1200mm, and any
    values outside this range are set to 0.
    """
    # dist = 0.1236 * np.tan((depth_img) / 2842.5 + 1.1863) * 1000-37
    dist = 0.075 * 580 / (1090 - disparity) * 8 * 1000
    dist[dist < 500] = 0
    dist[dist > 1200] = 0
    return dist


def plot_yolo_result(image: np.ndarray, results: np.ndarray) -> None:
    """
    Visualise the prediction result.

    Args:
        image (np.ndarray): The input image.
        results (np.ndarray): The prediction result.
    """
    # Draw rectangles for each detection result
    # Create figure and axis
    _, ax = plt.subplots()
    image = np.array(image)
    # Display the image
    ax.imshow(image)
    for i, detection in enumerate(results):
        draw_yolo_frame_plt(ax, detection, i + 1)
    # Show the plot
    plt.show()


def plot_sam_result(self, image: np.ndarray, result: np.ndarray) -> None:
    """
    Visualise the prediction result.

    Args:
        image (np.ndarray): The input image.
        result (np.ndarray): The prediction result.
    """
    image = np.array(image)
    contours_final = remove_small_cnt(result)
    new_mask = np.zeros_like(result)
    masks = cv2.drawContours(new_mask, contours_final, -1, (255, 255, 255), thickness=cv2.FILLED).astype(np.uint8)
    cv2.drawContours(masks, contours_final, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(image=image, contours=contours_final, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    cv2.bitwise_and(image, image, mask=masks)
    plt.imshow(image)
    plt.show()
