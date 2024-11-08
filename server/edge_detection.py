import cv2

IMAGE_FILENAME: str = "image.png"

image = cv2.imread(IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
_, binary_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)


edges = cv2.Canny(binary_img, 50, 150)
cv2.imwrite("edges.jpg", edges)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
wall_contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]

with open("contours.txt", "w", encoding="utf-8") as f:
    for contour in wall_contours:
        f.write(str(contour))
