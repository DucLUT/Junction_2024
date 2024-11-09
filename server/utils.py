"""
This module provides functions for extracting walls from floorplan images.
"""

import logging
from typing import List
import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess the image by converting it to grayscale, resizing, and applying Gaussian blur.

    Args:
        image (Image): The input image.

    Returns:
        np.ndarray: The preprocessed image.
    """
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Resize the image to make the lines thicker
    scale_percent = 200  # Percent of original size
    width = int(image_gray.shape[1] * scale_percent / 100)
    height = int(image_gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image_gray, dim, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    return blurred_image


def unwanted_features_mask(edges: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the edges in the image.
    """
    # Apply larger morphological kernel to clean up the image
    kernel = np.ones((7, 7), np.uint8)  # Increase kernel size for larger structures
    unwanted_features = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    unwanted_features = cv2.morphologyEx(unwanted_features, cv2.MORPH_OPEN, kernel)

    # Apply dilation followed by erosion with more iterations
    unwanted_features = cv2.dilate(
        unwanted_features, kernel, iterations=3
    )  # Increase iterations
    cv2.imwrite("unwanted_features_before_erode.jpg", unwanted_features)

    unwanted_features = cv2.erode(unwanted_features, kernel, iterations=2)

    cv2.imwrite("unwanted_features.jpg", unwanted_features)

    return unwanted_features


def get_wall_contours(cleaned_image: np.ndarray) -> List[np.ndarray]:
    """
    Get wall contours from the cleaned image.
    """
    # Find contours on the cleaned image
    contours, _ = cv2.findContours(
        cleaned_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,  # Use RETR_TREE to capture nested contours
    )

    wall_contours = []
    for cnt in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(cnt, 5, True)  # Loosen approximation tolerance

        # Calculate the bounding rectangle of the contour
        _, _, w, h = cv2.boundingRect(approx)

        # Calculate the aspect ratio and area of the bounding rectangle
        aspect_ratio = float(w) / h
        area = cv2.contourArea(approx)

        # Relax the filtering conditions for aspect ratio and area
        if (
            0.1 < aspect_ratio < 10.0 and area > 500
        ):  # Adjust thresholds for smaller walls
            wall_contours.append(approx)

    return wall_contours


def extract_walls(image_initial: Image) -> List[np.ndarray]:
    """
    Extract wall contours from the initial image.

    Args:
        image_initial (Image): The initial input image.

    Returns:
        List[np.ndarray]: A list of wall contours.
    """
    logger.info("Processing image...")
    # Preprocess the image
    preprocessed_image_array: np.ndarray = preprocess_image(image_initial)

    # Apply adaptive thresholding
    binary_img = cv2.adaptiveThreshold(
        preprocessed_image_array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Perform Canny edge detection
    edges = cv2.Canny(binary_img, 50, 150)
    cv2.imwrite("edges.jpg", edges)

    # Mask for unwanted features
    unwanted_features = unwanted_features_mask(edges)

    wall_only_mask = cv2.bitwise_not(unwanted_features)
    # Mask the walls in the edges image
    walls_extracted = cv2.bitwise_and(edges, edges, mask=wall_only_mask)
    cv2.imwrite("walls_extracted.jpg", walls_extracted)

    # Additional morphological operations to remove artifacts and non-wall features
    kernel = np.ones((3, 3), np.uint8)
    walls_extracted = cv2.morphologyEx(walls_extracted, cv2.MORPH_CLOSE, kernel)
    walls_extracted = cv2.morphologyEx(walls_extracted, cv2.MORPH_OPEN, kernel)

    wall_contours = get_wall_contours(walls_extracted)

    # Draw on blank image
    blank_image = np.zeros_like(walls_extracted)
    cv2.drawContours(blank_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("walls_contours.jpg", blank_image)

    return wall_contours
