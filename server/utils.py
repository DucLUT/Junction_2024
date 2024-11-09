"""
This module provides functions for extracting walls from floorplan images.
"""

import os
import logging
from typing import List,Tuple
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
    # Convert the image to grayscale if it is not already
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_array

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
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    scale_percent = 200  # Percent of original size
    width = int(image_gray.shape[1] * scale_percent / 100)
    height = int(image_gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image_gray, dim, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur to smooth the image
    preprocessed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    # Remove dotted lines
    cnts = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5000:
            cv2.drawContours(preprocessed_image, [c], -1, (0, 0, 0), -1)

    # Fill contours
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = 255 - cv2.morphologyEx(preprocessed_image, cv2.MORPH_CLOSE, close_kernel, iterations=6)
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15000:
            cv2.drawContours(close, [c], -1, (0, 0, 0), -1)

    # Smooth contours
    close = 255 - close
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, open_kernel, iterations=3)

    # Convert preprocessed_image to color
    preprocessed_image_color = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

    # Find contours and draw result
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(preprocessed_image_color, [c], -1, (36, 255, 12), 3)

    return preprocessed_image_color


def keep_straight_lines(image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Keep only the straight lines in the image using the Hough Line Transform.

    Args:
        image (np.ndarray): The input image.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int, int, int]]]: The image with only straight lines and the coordinates of the lines.
    """
    # Perform edge detection
    edges = cv2.Canny(image, 50, 150)

    # Use the Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )

    # Create a blank image to draw the lines
    line_image = np.zeros_like(edges)
    line_coordinates = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            line_coordinates.append((x1, y1, x2, y2))

    return line_image, line_coordinates


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
    Generate a new image of the same floorplan with only the walls using DALL-E.

    Args:
        image (Image): The input image.

    Returns:
        np.ndarray: The generated image as a numpy array.
    """
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("No OpenAI API key found")

    client = OpenAI(api_key=openai_api_key)

    size_x, size_y = image.size
    if size_x > 1024 or size_y > 1024:
        image.thumbnail((1024, 1024))

    # Ensure the size parameter is one of the allowed values
    allowed_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
    size = f"{image.size[0]}x{image.size[1]}"
    if size not in allowed_sizes:
        size = "1024x1024"

    response = client.images.generate(
        model="dall-e-3",
        prompt="Remove everything except for the walls from the image of the floorplan. Do not do anything else to the image, it should be completely the same as the original, except for only having the walls of the floorplan.",
        size=size,
        quality="standard",
        n=1,
    )

    image_url: str | None = response.data[0].url
    if image_url is None:
        raise ValueError("No image URL found in the response")

    return image_url


def remove_small_artifacts(image: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Remove small artifacts and noise from the image.

    Args:
        image (np.ndarray): The input image.
        min_size (int): Minimum size of artifacts to keep.

    Returns:
        np.ndarray: Image with small artifacts removed.
    """
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    cleaned_image = np.zeros(output.shape, dtype=np.uint8)
    for i in range(nb_components):
        if sizes[i] >= min_size:
            cleaned_image[output == i + 1] = 255

    return cleaned_image





def get_outer_contour(image: Image) -> np.ndarray:
    """
    Get the outer contour of all the walls, just the outer shell.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The image with the outer contour of all the walls.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform edge detection
    edges = cv2.Canny(preprocessed_image, 50, 150)

    # Remove artifacts
    cleaned_edges = remove_artifacts(edges, 7, 1)

    # Find the outer contour
    contours, _ = cv2.findContours(
        cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a blank image to draw the outer contour
    outer_contour_image = np.zeros_like(cleaned_edges)
    cv2.drawContours(outer_contour_image, contours, -1, (255, 255, 255), 2)

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

    # Ensure the image is in grayscale
    if len(preprocessed_image_array.shape) == 3 and preprocessed_image_array.shape[2] == 3:
        preprocessed_image_array = cv2.cvtColor(preprocessed_image_array, cv2.COLOR_RGB2GRAY)

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
    return wall_contours
