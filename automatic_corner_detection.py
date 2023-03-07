import numpy as np
import cv2

# Load image
cam = 1
img = cv2.imread(f"data/cam{cam}/calibration/test_1.png")

# Find and enhance the edges
edges = cv2.Canny(img, 200,300)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 1)
edges = cv2.erode(edges, kernel, iterations = 1)

# Find the contours and keep the largest (the chessboard contour)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
board = max(contours, key = cv2.contourArea)

# Find only the squares
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh_inv = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
stencil = np.ones(gray.shape).astype(gray.dtype)*255
blank = np.zeros(gray.shape).astype(gray.dtype)
color = [0]
cv2.fillPoly(stencil, [board], color)
mask = stencil+thresh_inv

edges = cv2.Canny(mask, 10, 50)
edges = cv2.dilate(edges, kernel)
edges = cv2.erode(edges, kernel)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
board = max(contours, key = cv2.contourArea)
cv2.drawContours(blank, [board], -1, 255, -1)
blank = cv2.erode(blank, kernel)

chessboard = cv2.bitwise_and(blank, mask)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()