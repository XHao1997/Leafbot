from __future__ import annotations
import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH
)
sys.path.append(SOURCE_PATH)
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_process import convert_to_xyxy, remove_small_cnt, draw_yolo_frame_plt
from pathlib import Path
from onnxruntime import InferenceSession
from yolonnx.services import Detector
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy
from utils import image_process
import PIL.Image


class Yolo():
    """
    YOLO model for object detection.
    """

    def __init__(self):
        """
        Initialize the YOLO model.
        """

        model = Path("weights/best.onnx")
        session = InferenceSession(
            model.as_posix(),
            providers=[
                "AzureExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        predictor = Detector(session, PillowToTensorContainStrategy(), conf_threshold=0.5)
        self.model = predictor.run

    def predict(self, image: np.ndarray) -> (list, np.array):
        """
        Make predictions using the YOLO model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The prediction result.
        """
        image_yolo = PIL.Image.fromarray(image)
        results = self.model(image_yolo)
        return results

    @staticmethod
    def visualise_result(image: np.ndarray, results: np.ndarray) -> None:
        """
        Visualise the prediction result.

        Args:
            image (np.ndarray): The input image.
            results (np.ndarray): The prediction result.
        """
        # Draw rectangles for each detection result
        # Create figure and axis
        fig, ax = plt.subplots()
        image = np.array(image)
        # Display the image
        ax.imshow(image)
        for i, detection in enumerate(results):
            draw_yolo_frame_plt(ax, detection, i + 1)
        # Show the plot
        plt.show()

    @staticmethod
    def get_image_with_bbox(image: np.ndarray, results: np):
        return image_process.draw_yolo_frame_cv(image, results)


class MobileSAM:
    """
    Mobile SAM model for image segmentation.
    """

    def __init__(self):
        """
        Initialize the Mobile SAM model.
        """
        model_type = "vit_t"
        sam_checkpoint = "weights/mobile_sam1.pt"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model = SamPredictor(mobile_sam)

    def predict(self, image: np.ndarray, yolo_results: list):
        """
        Make predictions using the Mobile SAM model.

        Args:
            image (np.ndarray): The input image.
            yolo_results (np.ndarray): Results from YOLO model.

        Returns:
            np.ndarray: The prediction result.
        """
        # Store xyxy bounding boxes in a list
        print("segment anything!!!\n")
        image = np.array(image)
        self.model.set_image(image)
        bbox_list = np.asarray([convert_to_xyxy(result) for result in yolo_results])
        centers = np.zeros((bbox_list.shape[0], 2))
        for i, box in enumerate(bbox_list):
            center_x, center_y = box[0] / 2 + box[2] / 2, box[1] / 2 + box[3] / 2
            centers[i, :] = np.array([center_x, center_y])
        for i, center in enumerate(centers):
            masks, scores, logits = self.model.predict(
                point_coords=center.reshape(1, 2),
                box=bbox_list[i],
                point_labels=[1],
                multimask_output=False)
            masks = (np.moveaxis(masks, 0, -1)).astype(np.uint8)
            best_mask = masks[:, :, np.argmax(scores)]
            if i == 0:
                masks_final = best_mask
            else:
                masks_final += best_mask

        contours_final = remove_small_cnt(masks_final)
        result = np.zeros_like(masks_final)
        result = cv2.drawContours(result, contours_final, -1, (255, 255, 255),
                                  thickness=-1).astype(np.uint8)
        return result

    @staticmethod
    def visualise_result(image: np.ndarray, result: np.ndarray) -> None:
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
