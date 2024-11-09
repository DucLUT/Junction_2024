"""
This module provides functions for extracting walls from floorplan images.
"""

import logging
from typing import List
import cv2
import numpy as np
from PIL import Image
import fitz
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_pdf_layers(pdf_bytes: bytes) -> List:
    # Open the PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    logger.info("DOC: %s", str(doc))
    layers: List = doc.get_layers()
    for layer in layers:
        logger.info("Layer name: %s, visibility: %s", layer[0], layer[1])

    if len(layers) == 0:
        logger.warning("No layers found in the PDF.")

    ocgs: dict = doc.get_ocgs()
    for ocg in ocgs:
        logger.info("OCG name: %s, state: %s", ocg[0], ocg[1])

    if len(ocgs) == 0:
        logger.warning("No OCGs found in the PDF.")

    doc.close()

    return layers


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
def process_contours(image: np.ndarray) -> np.ndarray:
    """
    Process the preprocessed image to remove dotted lines, fill contours, smooth contours,
    and draw the final contours on the image.

    Args:
        image (np.ndarray): The preprocessed image.

    Returns:
        np.ndarray: The image with the final contours drawn.
    """
    # Convert the image to grayscale and resize
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    scale_percent = 200  # Adjust as needed
    width = int(image_gray.shape[1] * scale_percent / 100)
    height = int(image_gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image_gray, dim, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred_image, 50, 150)
    cv2.imwrite("edges_debug.jpg", edges)  # Debug output

    # Remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw contours
    contour_image = np.zeros_like(cleaned_edges)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Use an appropriate threshold for your image scale
            # Draw only significant contours
            cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Smooth the contours
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    smoothed_contours = cv2.morphologyEx(contour_image, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)

    # Convert to color for final display
    final_image = cv2.cvtColor(smoothed_contours, cv2.COLOR_GRAY2BGR)

    # Draw contours on the original image
    contours_final, _ = cv2.findContours(smoothed_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_final:
        cv2.drawContours(final_image, [contour], -1, (0, 255, 0), 3)

    return final_image



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


# def get_outer_contour(image: Image) -> np.ndarray:
#     """
#     Get the outer contour of all the walls, just the outer shell.

#     Args:
#         image (np.ndarray): The input image.

#     Returns:
#         np.ndarray: The image with the outer contour of all the walls.
#     """
#     # Preprocess the image
#     preprocessed_image = preprocess_image(image)

#     # Perform edge detection
#     edges = cv2.Canny(preprocessed_image, 50, 150)

#     # Remove artifacts
#     cleaned_edges = remove_artifacts(edges, 7, 1)

#     # Find the outer contour
#     contours, _ = cv2.findContours(
#         cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # Create a blank image to draw the outer contour
#     outer_contour_image = np.zeros_like(cleaned_edges)
#     cv2.drawContours(outer_contour_image, contours, -1, (255, 255, 255), 2)

#     return outer_contour_image

#orginal get_outer_contour
def get_outer_contour(edges: np.ndarray) -> np.ndarray:
    """
    Extracts the outermost contour from the precomputed edges by combining all detected wall edges.

    Args:
        edges (np.ndarray): An edge-detected binary image where walls are represented
                            by white pixels (255) on a black background (0).

    Returns:
        np.ndarray: An image containing only the outer contour of the floorplan.
    """
    # Ensure edges is a numpy array
    if not isinstance(edges, np.ndarray):
        raise ValueError("The 'edges' parameter must be a numpy array.")

    # Step 1: Remove small artifacts from the edges using morphological operations
    cleaned_edges = remove_artifacts(edges, size=7, iterations=2)

    # Step 2: Find all contours in the cleaned edge image
    contours, _ = cv2.findContours(
        cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 3: Concatenate all contour points into a single array to form a combined outline
    all_points = np.vstack(contours)

    # Step 4: Compute the convex hull of the combined points to get a single outer boundary
    hull = cv2.convexHull(all_points)

    # Step 5: Draw the outer contour on a blank image
    outer_contour_image = np.zeros_like(edges)
    cv2.drawContours(outer_contour_image, [hull], -1, (255, 255, 255), thickness=2)

    return outer_contour_image


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes







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

    outer_contour = get_outer_contour(edges)
    cv2.imwrite("outer_contour.jpg", outer_contour)
    logger.info("Outer contour drawn. Image written to outer_contour.jpg")

    # Draw on blank image
    blank_image = np.zeros_like(cleaned_image)
    cv2.drawContours(blank_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("walls_contours.jpg", blank_image)
    logger.info("Contours drawn on blank image. Image written to walls_contours.jpg")

    return wall_contours
