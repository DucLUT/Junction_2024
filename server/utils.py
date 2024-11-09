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


def remove_artifacts(edges: np.ndarray, size: int, iterations: int = 1) -> np.ndarray:
    """
    Remove artifacts from the image using morphological operations.
    """
    kernel = np.ones((size, size), np.uint8)
    artifact_filter = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )
    artifact_filter = cv2.morphologyEx(
        artifact_filter, cv2.MORPH_OPEN, kernel, iterations=iterations
    )

    artifact_filter = cv2.dilate(artifact_filter, kernel, iterations=3)
    artifact_filter = cv2.erode(artifact_filter, kernel, iterations=2)

    cleaned_image = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(artifact_filter))
    return cleaned_image


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


def draw_outer_contour(edges: np.ndarray) -> np.ndarray:
    """
    Draw the outer contour of the image.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contour = np.zeros_like(edges)
    cv2.drawContours(outer_contour, contours, -1, (255, 255, 255), 2)
    return outer_contour


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
    logger.info("Image preprocessed.")

    # Apply adaptive thresholding
    binary_img = cv2.adaptiveThreshold(
        preprocessed_image_array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    logger.info("Adaptive thresholding applied.")

    # Perform Canny edge detection
    edges = cv2.Canny(binary_img, 50, 150)
    cv2.imwrite("edges.jpg", edges)
    logger.info("Canny edge detection performed. Image written to edges.jpg")

    # Clean unwanted features
    cleaned_image = remove_artifacts(edges, 7, 1)
    cv2.imwrite("cleaned_image_1.jpg", cleaned_image)
    logger.info(
        "First artifact removal step completed. Image written to cleaned_image_1.jpg"
    )

    artifact_removal_step: int = 2
    while artifact_removal_step < 4:
        size = 3
        cleaned_image = remove_artifacts(cleaned_image, size, artifact_removal_step)
        cv2.imwrite(f"cleaned_image_{artifact_removal_step}.jpg", cleaned_image)
        logger.info(
            "Artifact removal step %d completed. Image written to cleaned_image_%d.jpg",
            artifact_removal_step,
            artifact_removal_step,
        )
        artifact_removal_step += 1

    wall_contours = get_wall_contours(cleaned_image)
    logger.info("Wall contours extracted.")

    outer_contour = draw_outer_contour(edges)
    cv2.imwrite("outer_contour.jpg", outer_contour)

    # Draw on blank image
    blank_image = np.zeros_like(cleaned_image)
    cv2.drawContours(blank_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("walls_contours.jpg", blank_image)
    logger.info("Contours drawn on blank image. Image written to walls_contours.jpg")

    return wall_contours
