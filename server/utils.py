import cv2
import numpy as np
from PIL import Image
from typing import List
import logging
import keras_ocr
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image: Image) -> np.ndarray:
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

    # Save the preprocessed image
    cv2.imwrite("preprocessed_image.jpg", blurred_image)

    return blurred_image


def get_cleaned_edges(edges: np.ndarray) -> np.ndarray:
    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_OPEN, kernel)

    # Write the cleaned edges to a file
    cv2.imwrite("cleaned_edges.jpg", cleaned_edges)

    return cleaned_edges


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


# Pytesseract implementation
# def inpaint_text(image: np.ndarray) -> np.ndarray:
#     # Convert image to RGB for pytesseract
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     # Use pytesseract to get bounding boxes for text
#     data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)

#     # Initialize a mask for inpainting
#     mask = np.zeros(image.shape[:2], dtype="uint8")

#     # Loop through each detected text box and create a mask
#     for i in range(len(data['text'])):
#         if int(data['conf'][i]) > 60:  # Confidence threshold, adjust as needed
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # Fill the mask at text location

#     # Inpaint the text regions using the mask
#     inpainted_image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
#     cv2.imwrite("inpaint_text.jpg", inpainted_image)

#     return inpainted_image


def inpaint_text(
    image: np.ndarray, pipeline: keras_ocr.pipeline.Pipeline
) -> np.ndarray:
    prediction_groups = pipeline.recognize([image])
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

    inpainted_image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    cv2.imwrite("inpaint_text.jpg", inpainted_image)
    return inpainted_image


def get_components(image: np.ndarray) -> np.ndarray:
    """
    Extracts connected components from the image and filters them based on size.

    Args:
        image (np.ndarray): The input binary image.

    Returns:
        np.ndarray: The output image with filtered components.
    """
    logger.info("Getting connected components...")
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    # connectedComponentsWithStats yields every separated component with information
    # on each of them, such as size.
    # The following part is just taking out the background which is also considered
    # a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # Minimum size of particles we want to keep (number of pixels).
    # Here, it's a fixed value, but you can set it as you want, e.g., the mean of the sizes or whatever.
    min_size = 150

    # Create a mask for components that are above the min_size
    logger.info("Filtering components with size >= %d...", min_size)
    logger.info("Number of components: %d", nb_components)
    mask = np.zeros(output.shape, dtype=np.uint8)
    for i in range(1, nb_components + 1):
        if i % 200 == 0:
            logger.info("Component %d: size = %d", i, sizes[i - 1])
        if sizes[i - 1] >= min_size:
            mask[output == i] = 255

    # Write the output image to a file.
    logger.info("Writing output image to file...")
    cv2.imwrite("get_components.jpg", mask)

    return mask


def get_wall_contours(image_initial: Image) -> List[np.ndarray]:
    pipeline: keras_ocr.pipeline.Pipeline = keras_ocr.pipeline.Pipeline()

    logger.info("Processing image...")
    # Preprocess the image
    preprocessed_image_array: np.ndarray = preprocess_image(image_initial)

    # Apply binary thresholding
    _, binary_img = cv2.threshold(preprocessed_image_array, 128, 255, cv2.THRESH_BINARY)

    # Perform Canny edge detection
    edges = cv2.Canny(binary_img, 50, 150)
    cv2.imwrite("edges.jpg", edges)

    inpaint_text(preprocessed_image_array, pipeline)
    # get_components(binary_img)

    # Clean up the edges
    cleaned_edges = get_cleaned_edges(edges)

    # Use Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(
        cleaned_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    # Create a blank image to draw the lines
    line_image = np.zeros_like(cleaned_edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

    # Find contours on the line image
    contours, _ = cv2.findContours(
        line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    wall_contours = []
    for cnt in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(cnt, 3, True)

        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(approx)

        # Calculate the aspect ratio and area of the bounding rectangle
        aspect_ratio = float(w) / h
        area = cv2.contourArea(approx)

        # Filter contours based on aspect ratio and area
        if 0.2 < aspect_ratio < 5.0 and area > 1000:
            wall_contours.append(approx)

    # Create a blank image to draw the contours
    contour_image = np.zeros_like(preprocessed_image_array)
    cv2.drawContours(contour_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("wall_contours.jpg", contour_image)

    return wall_contours
