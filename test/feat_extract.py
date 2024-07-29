import numpy as np
import cv2
import matplotlib.pyplot as plt
import supervision as sv
import skimage
from ultralytics import YOLOv10


def bbox2(img):
    # from https://stackoverflow.com/a/31402351/19249364
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def create_mask_from_bbox(image_shape, bbox):
    """
    Create a binary mask from a bounding box.

    Args:
    image_shape (tuple): Shape of the image (height, width).
    bbox (tuple): Bounding box in (x1, y1, x2, y2) format.

    Returns:
    np.array: Binary mask with the bounding box area set to 1.
    """
    # Initialize a blank mask
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

    # Unpack bounding box coordinates
    x1, y1, x2, y2 = bbox[0].astype(np.int32)
    # Set the bounding box area in the mask to 1

    mask[y1 - 5:y2, x1:x2] = 1

    return mask


model_gripper = YOLOv10('../weights/best_gripper.pt')
img = np.load('../data/cali/j1/joint1_nn_01.npy')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.stack((img,) * 3, axis=-1)

results = model_gripper(source=img, conf=0.25)[0]
detections = sv.Detections.from_ultralytics(results)
mask = create_mask_from_bbox(img.shape, detections.xyxy)
gripper = cv2.bitwise_and(img, img, mask=mask)

bbox = bbox2(gripper)
cropped = gripper[bbox[0] - 1:bbox[1] + 1, bbox[2] - 1:bbox[3] + 1, :]
fd, hog_img = skimage.feature.hog(cropped[:, :, 0], orientations=9, pixels_per_cell=(4, 4),
                                  cells_per_block=(4, 4), visualize=True)
hog_img = cv2.resize(hog_img, (64, 64))
sv.plot_image(hog_img)
