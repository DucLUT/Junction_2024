import cv2
import numpy as np
from PIL import Image
from typing import List


def get_wall_contours(image_initial: Image) -> List[np.ndarray]:
    # Convert the image to grayscale
    image = cv2.cvtColor(np.array(image_initial), cv2.COLOR_RGB2GRAY)

    # Resize the image to make the lines thicker
    scale_percent = 200  # Percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Apply binary thresholding
    _, binary_img = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Perform Canny edge detection
    edges = cv2.Canny(binary_img, 50, 150)
    cv2.imwrite("edges.jpg", edges)

    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(cleaned_edges, cv2.MORPH_OPEN, kernel)

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
    contour_image = np.zeros_like(resized_image)
    cv2.drawContours(contour_image, wall_contours, -1, (255, 255, 255), 2)
    cv2.imwrite("wall_contours.jpg", contour_image)

    return wall_contours
