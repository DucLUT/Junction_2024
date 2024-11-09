import cv2
import numpy as np
from PIL import Image
import logging
from typing import List

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


def get_cleaned_edges(edges: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the edges in the image.

    Args:
        edges (np.ndarray): The input edges.

    Returns:
        np.ndarray: The cleaned edges.
    """
    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_OPEN, kernel)

    # Apply dilation followed by erosion to preserve larger structures (walls)
    cleaned_edges = cv2.dilate(cleaned_edges, kernel, iterations=2)
    cleaned_edges = cv2.erode(cleaned_edges, kernel, iterations=2)

    cv2.imwrite("cleaned_edges.jpg", cleaned_edges)

    return cleaned_edges


def get_wall_contours(cleaned_image: np.ndarray) -> List[np.ndarray]:
    """
    Get wall contours from the cleaned image.

    Args:
        cleaned_image (np.ndarray): The cleaned image.

    Returns:
        List[np.ndarray]: A list of wall contours.
    """
    # Find contours on the cleaned image
    contours, _ = cv2.findContours(
        cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    wall_contours = []
    for cnt in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(cnt, 3, True)

        # Calculate the bounding rectangle of the contour
        _, _, w, h = cv2.boundingRect(approx)

        # Calculate the aspect ratio and area of the bounding rectangle
        aspect_ratio = float(w) / h
        area = cv2.contourArea(approx)

        # Filter contours based on aspect ratio and area
        if 0.2 < aspect_ratio < 5.0 and area > 1000:
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

    # Clean up the edges
    cleaned_edges = get_cleaned_edges(edges)

    wall_only_mask = cv2.bitwise_not(cleaned_edges)
    # Mask the walls in the edges image
    walls_extracted = cv2.bitwise_and(edges, edges, mask=wall_only_mask)
    cv2.imwrite("walls_extracted.jpg", walls_extracted)

    wall_contours = get_wall_contours(walls_extracted)

    # Draw on blank image
    blank_image = np.zeros_like(walls_extracted)
    cv2.drawContours(blank_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("walls_contours.jpg", blank_image)

    return wall_contours
