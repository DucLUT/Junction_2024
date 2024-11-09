import cv2
import numpy as np
from PIL import Image


def get_contours(image_initial: Image) -> list[np.ndarray]:
    image = cv2.cvtColor(np.array(image_initial), cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary_img, 50, 150)
    cv2.imwrite("edges.jpg", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]

    return wall_contours
def draw_contours(image_initial: Image, contours: list[np.ndarray]) -> Image:
    # Create a blank image for drawing contours
    image = np.array(image_initial)
    blank = np.zeros_like(image)

    # Draw the filtered contours on the blank image
    cv2.drawContours(blank, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return Image.fromarray(blank)

input_image = Image.open('image.png')
contours = get_contours(input_image)

# Draw the contours on a blank image to remove smaller elements
output_image = draw_contours(input_image, contours)

# Save or display the final image
output_image.save('processed image.png')
output_image.show()