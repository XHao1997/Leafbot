import numpy as np
import cv2
import copy


def get_cnts(mask):
    """
    The function `get_cnts` takes a mask image as input, finds contours in the mask image using OpenCV,
    and returns the contours.
    
    Args:
        mask: The `mask` parameter in the `get_cnts` function is likely a binary image or a mask image
    that is used for finding contours.Contours are the boundaries of objects in an image, and they are
    useful for various image processing tasks such as object detection, shape analysis, and image
    segmentation.
    
    Returns:
        Contours of the mask are being returned.
    """
    mask_copy = copy.deepcopy(mask)
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_optimal_picking_points(contours, corner_list):
    optimal_points = []
    for cnt, corner in zip(contours, corner_list):
        cnt = cnt.reshape(-1, 2)
        corner = np.asarray(corner).reshape(-1, 2)
        corners_left = corner[np.argmin(corner[:, 0])]
        corners_down = corner[np.argmax(corner[:, 1])]
        optimal_points.append(cnt[np.argmin(np.linalg.norm(cnt - corners_left, axis=1))])
        optimal_points.append(cnt[np.argmin(np.linalg.norm(cnt - corners_down, axis=1))])
    optimal_points = np.asarray(optimal_points).reshape(-1, 2)
    return optimal_points


def get_incircle(mask, cnts):
    """
    The `get_incircle` function calculates the maximum distance from the centroid of a contour to any
    point on the contour and draws a circle with that radius for each contour provided.
    
    Args:
        mask: The `mask` parameter is a NumPy array representing an image mask. It is used to create a new
    mask that will contain the inscribed circles for each contour provided in the `cnts` parameter.
        cnts: The `cnts` parameter in the `get_incircle` function is typically a list of contours.
    Contours are essentially a list of points that represent a shape or object in an image. Each contour
    is a numpy array of shape `(N, 1, 2)`, where `N
    
    Returns:
        The `get_incircle` function returns two values:
    1. `mask`: A NumPy array representing an image mask that contains the inscribed circles for each
    contour provided in the `cnts` parameter.
    2. `(cx, cy)`: A tuple containing the coordinates of the centroid of the last contour processed in
    the loop.
    """
    mask = np.zeros_like(mask).astype(np.uint8)
    cx, cy = (None, None)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            diff = (cnt - [cx, cy]).reshape(-1, 2)
            # Calculate Euclidean distance between each point in the contour and center1
            distances = np.linalg.norm(diff, axis=1)
            max_dist = min(distances)
            cv2.circle(mask, (cx, cy), int(max_dist *0.8), (255, 255, 255), -1)
    return mask.astype(np.uint8), (cx, cy)


def find_furthest_points(cnt):
    """
    The function `find_furthest_points` calculates the two points in a given set that are furthest
    apart.
    
    Args:
        cnt: It seems like the parameter `cnt` is expected to be a collection of points. The function
    `find_furthest_points` is designed to find the pair of points in the collection `cnt` that are
    furthest apart from each other. The function calculates the distance between all pairs of points and
    
    Returns:
        The function `find_furthest_points` returns a list containing the indices of the two points in the
    input `cnt` list that are furthest apart from each other.
    """
    max_dist_prev = 0
    max_index = [None, None]
    for i, points in enumerate(cnt):
        max_dist = np.max(np.linalg.norm(cnt - points, axis=1))
        if max_dist > max_dist_prev:
            max_index = [i, np.argmax(np.linalg.norm(cnt - points, axis=1))]
        max_dist_prev = max_dist
    return max_index


def find_midpoint(point1, point2):
    """
    The function `find_midpoint` calculates the midpoint between two points in a two-dimensional space.
    
    Args:
        point1: It seems like you were about to provide the details of the `point1` parameter but the
    message got cut off. Could you please provide the details of the `point1` parameter so that I can
    assist you further?
        point2: It seems like you have provided the code snippet for finding the midpoint between two
    points, but you have not provided the definition or values for `point2`. In order to help you find
    the midpoint, I would need the values for `point2`. Could you please provide the values for `point2
    
    Returns:
        The function `find_midpoint` takes two points as input, flattens them using `ravel()`, calculates
    the midpoint between the two points, and returns the coordinates of the midpoint as a NumPy array.
    :param point1:
    :param point2:
    """
    point1 = point1.ravel()
    point2 = point2.ravel()
    return np.array([(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2])


def find_y_intercept(mid_point, slope):
    """
    The function calculates the y-intercept of a line given its midpoint and slope.
    
    Args:
        mid_point: The `mid_point` parameter represents the coordinates of the midpoint of a line segment.
    It is typically a tuple containing the x and y coordinates of the midpoint. For example, `mid_point
    = (3, 5)` would represent a midpoint with x-coordinate 3 and y-coordinate 5.
        slope: The slope parameter represents the slope of a line, which is the ratio of the vertical
    change (rise) to the horizontal change (run) between two points on the line. It determines the
    steepness of the line.
    
    Returns:
        The function `find_y_intercept` returns the y-intercept of a line given the midpoint and slope of
    the line.
    """
    return mid_point[1] - slope * mid_point[0]


def calculate_slope(point1, point2):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')
    return -1 / slope


def distance_to_line(x0, y0, a, b):
    """
    The function calculates the perpendicular distance from a point (x0, y0) to a line defined by the
    equation y = ax + b.
    
    Args:
        x0: The parameter `x0` represents the x-coordinate of a point in a 2D plane.
        y0: The parameter `y0` represents the y-coordinate of a point in a 2D plane.
        a: The parameter `a` in the `distance_to_line` function represents the slope of the line. It is
    used in the formula to calculate the distance from a point `(x0, y0)` to the line defined by the
    equation `y = ax + b`.
        b: The parameter `b` in the `distance_to_line` function represents the y-intercept of the line in
    the form of `y = ax + b`, where `a` is the slope of the line.
    
    Returns:
        The function `distance_to_line` returns the perpendicular distance from the point (x0, y0) to the
    line defined by the equation y = ax + b.
    """
    return abs(a * x0 - y0 + b) / np.sqrt(a ** 2 + 1)


def get_leaf_corner(leaf_contour):
    """
    This function takes a leaf contour as input, finds the furthest points, calculates the midpoint, and
    then identifies the leaf corners based on distance to a line.
    
    Args:
        leaf_contour: It seems like you have provided the code snippet without the actual input for the
    `leaf_contour` parameter. Could you please provide the actual input data for `leaf_contour` so that
    I can assist you further with running the function `get_leaf_corner`?
    
    Returns:
        The function `get_leaf_corner` returns five points: `point1`, `point2`, `point3`, `point4`, and
    `midpoint`.
    """
    point1, point2 = leaf_contour[find_furthest_points(leaf_contour)]
    midpoint = find_midpoint(point1, point2)
    slope = calculate_slope(point1, point2)
    intercept = find_y_intercept(midpoint.astype(int), slope)
    dist_list = np.zeros(leaf_contour.shape[0])
    for index, p in enumerate(leaf_contour):
        dist_list[index] = distance_to_line(p[0], p[1], slope, intercept)
    result = dist_list
    smallest_index = np.argpartition(abs(result), 5)[:5]
    point3 = leaf_contour[smallest_index[0]]
    point4_index = np.argmin(np.linalg.norm((leaf_contour[smallest_index] + point3 - 2 * midpoint), axis=1))
    point4 = leaf_contour[smallest_index[point4_index]]
    return point1, point2, point3, point4, midpoint


def draw_leaf_corner(corners, mask):
    """
    The function `draw_leaf_corner` takes a list of corner points and a mask image, then draws red
    circles at each corner point on a new mask image.
    
    Args:
        corners: The `corners` parameter is a list of points that represent the corners of a leaf in an
    image. Each point is a tuple containing the x and y coordinates of the corner.
        mask: The `mask` parameter is likely a binary image or a mask image that is being used to draw
    leaf corners on top of it. The function `draw_leaf_corner` takes a list of corner points (`corners`)
    and a mask image (`mask`) as input. It then creates a new mask
    
    Returns:
        The function `draw_leaf_corner` returns a mask with red circles drawn at the specified corner
    points.
    """
    mask1 = np.zeros_like(copy.deepcopy(mask))
    for point in corners:
        cv2.circle(mask1, point.astype(int), 5, (255, 0, 255), -1)  # Red point
    return mask1


def get_leaf_center(mask, leaf_index=-1):
    """
    This function extracts the center coordinates of leaf shapes from a given mask image.
    
    Args:
        mask: The `mask` parameter is likely a binary image representing the shape of a leaf, where the
    leaf pixels are typically white (pixel value of 255) and the background is black (pixel value of 0).
    This binary mask is used to identify the contours of the leaf for further processing in the
        leaf_index: The `leaf_index` parameter is used to specify a particular leaf contour to extract the
    center of. By default, it is set to -1, which means that if no specific leaf index is provided, the
    function will calculate the center for all leaf contours found in the mask image. If a specific
    
    Returns:
        A list of leaf centers is being returned. If a specific leaf index is provided, the function will
    return the center of that particular leaf. Otherwise, it will return the centers of all the leaves
    detected in the mask.
    """
    contours = get_cnts(mask)
    leaf_centers = []
    if leaf_index != -1:
        cnt = contours[leaf_index]
        cnt = cnt.reshape(-1, 2)
        leaf_centers.append(get_leaf_corner(cnt)[-1])
    else:
        for cnt in contours:
            for cnt in contours:
                cnt = cnt.reshape(-1, 2)
                leaf_center = get_leaf_corner(cnt)[-1]
                leaf_centers.append(leaf_center)
    return leaf_centers


def draw_circle(point, image, size=5):
    """
    The function `draw_circle` takes a point and an image as input, and draws a filled circle of a
    specified size at that point on the image.
    
    Args:
        point: The `point` parameter is a numpy array representing the coordinates of the center of the
    circle to be drawn on the `image`. The `point` array should have two elements, where the first
    element represents the x-coordinate and the second element represents the y-coordinate of the center
    of the circle.
        image: The `image` parameter is the image on which you want to draw a circle. It is typically a
    NumPy array representing an image where the circle will be drawn.
        size: The `size` parameter in the `draw_circle` function represents the radius of the circle that
    will be drawn on the `image`. The default value for `size` is 5, which means that the circle will
    have a radius of 5 pixels if no other value is provided when calling the. Defaults to 5
    
    Returns:
    The function `draw_circle` returns the image with a circle drawn on it at the specified point with
    the specified size.
    """
    x, y = point.astype(np.int16)
    cv2.circle(image, (x, y), int(size), (255, 255, 255), thickness=-1)
    return image
