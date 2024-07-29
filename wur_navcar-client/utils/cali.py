import cv2
import numpy as np
import glob
import supervision as sv
from sklearn.decomposition import PCA

from utils.image_process import convert_to_xyxy


def sort_file_by_num(img_list, index=13):
    img_list.sort()
    img_list.sort(key=lambda x: int(x[index:-4]))  ## sort by num
    img_nums = len(img_list)
    for i in range(img_nums):
        img_name = img_list[i]


def read_all_files_by_name(filename_pattern, option='rgb'):
    # Get a list of image paths matching the pattern
    img_list = glob.glob(filename_pattern)
    # Read all images and store them in a list
    if option != 'rgb':
        sort_file_by_num(img_list, 17)
        images = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in img_list]
    else:
        sort_file_by_num(img_list)

        images = [cv2.imread(img_path).astype(np.uint8) for img_path in img_list]

    # Filter out any None values that occur if an image fails to load
    images = [img for img in images if img is not None]
    return np.array(images), img_list


def expand_mask_roi(mask, scale_factor=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    mask_region = mask[y:y + h, x:x + w]
    mask_resized = cv2.resize(mask_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    new_mask = np.zeros_like(mask)
    new_x, new_y = max(0, x - (new_w - w) // 2), max(0, y - (new_h - h) // 2)
    try:
        new_mask[new_y:new_y + new_h, new_x:new_x + new_w] = mask_resized
    except:
        new_mask = mask
    return new_mask


def detect_tag_corner(img):
    # Define the dictionary and parameters for detection
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    params = cv2.aruco.DetectorParameters()
    # Create the detector
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    #     plt.imshow(img)
    #     plt.show()
    return marker_corners[0]


def get_incircle(img):
    # Find contours in the mask
    mask_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize a distance matrix
    raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
    # Calculate the distance of each point from the contour
    for i in range(mask_gray.shape[0]):
        for j in range(mask_gray.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)
    # Find the maximum distance point
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    # Get the absolute value of the maximum distance
    maxVal = abs(maxVal)
    # Calculate the radius of the incircle
    radius = int(maxVal)
    # Create an empty mask
    result = np.zeros_like(mask_gray)
    # Draw the filled incircle
    cv2.circle(result, maxDistPt, int(0.8 * radius), (255, 255, 255), thickness=-1)
    return result


def preprocess(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.stack((img,) * 3, axis=-1)

    results = model(source=img, conf=0.1, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    mask = create_mask_from_bbox(img.shape, detections.xyxy)
    gripper = cv2.bitwise_and(img, img, mask=mask)

    bbox = bbox2(gripper)
    downside = int((bbox[0] + bbox[1]) / 2) + 5
    cropped = gripper[bbox[0] - 1:downside + 1, bbox[2]:bbox[3] + 1, :]
    cropped = cv2.resize(cropped, (64, 64))
    return cropped[:, :, 0]


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


def bbox2(img):
    # from https://stackoverflow.com/a/31402351/19249364
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def segment_leaf_sam(img, yolo_result, server):
    mask_leaves = server.sam_mask(img)
    bbox_xyxy = np.array([convert_to_xyxy(yolo_result)])
    mask = create_mask_from_bbox(img.shape, bbox_xyxy)
    mask = cv2.bitwise_or(mask, mask, mask=mask_leaves)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    #     Apply Gaussian blur to smooth the edges
    mask_cleaned = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)

    return mask_cleaned


def pca_pick_angle(img, yolo_results, server, index=-1):
    yolo_result = yolo_results[index]
    mask_cleaned = segment_leaf_sam(img, yolo_result, server)
    # Convert mask to coordinates
    #     mask_cleaned = (sam.predict(img, [yolo_result]))

    y_coords, x_coords = np.where(mask_cleaned > 0.1)
    coordinates = list(zip(x_coords, y_coords))
    X = 480 - y_coords
    y = x_coords
    X = np.vstack((X, y)).T
    pca = PCA(n_components=1)
    pca.fit_transform(X)
    interp = pca.components_[0][0] / pca.components_[0][1]
    return interp


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
    print(bbox)
    # Unpack bounding box coordinates
    x1, y1, x2, y2 = bbox[0].astype(np.int32)
    # Set the bounding box area in the mask to 1

    mask[y1 - 5:y2, x1:x2] = 1

    return mask
