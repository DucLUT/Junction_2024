import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Convert PDF to Image
def convert_pdf_to_image(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    return pages[0]  # Assuming the floor plan is on the first page

# Process Floor Plan Image
def process_floor_plan(image: Image, output_path: str):
    # Convert PIL Image to OpenCV format
    image = np.array(image.convert('L'))  # Convert to grayscale

    # Apply binary threshold to create a binary image
    _, binary_img = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)

    # Use morphological closing to fill small gaps in the lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close)

    # Apply morphological opening to remove thinner, isolated lines (dotted lines)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(opened, 30, 120)

    # Use Hough Line Transformation to detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    # Create a blank image to draw the main structure lines
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > 100:  # Filter out shorter lines
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

    # Combine the line image with the opened image to keep the main structure
    main_structure = cv2.bitwise_and(line_image, opened)

    # Find contours in the filtered image to remove smaller elements
    contours, _ = cv2.findContours(main_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to keep only large contours
    mask = np.zeros_like(main_structure)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Increase the area threshold for further filtering
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to isolate the main building structure
    final_output = cv2.bitwise_and(main_structure, mask)

    # Save the final processed image
    cv2.imwrite(output_path, final_output)
    Image.fromarray(final_output).show()

# Paths to the PDF file and output image file
pdf_path = '/Users/qimengshi/Documents/GitHub/Junction_2024/SiteMaterial/Material to share/Site 1/floor_6.pdf'
output_image_path = '/Users/qimengshi/Documents/GitHub/Junction_2024/SiteMaterial/Material to share/Site 1/processed_floor_plan.png'

# Convert PDF to image and process
image = convert_pdf_to_image(pdf_path)
process_floor_plan(image, output_image_path)
