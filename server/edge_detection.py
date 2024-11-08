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
