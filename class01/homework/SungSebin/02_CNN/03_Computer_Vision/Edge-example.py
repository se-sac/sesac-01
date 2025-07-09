import cv2
import numpy as np

img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
#kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
identity = np.array([[0,0,0],[0,1,0],[0,0,0]]) 
ridge = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) 
edge_detection = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) 
sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
box_blur = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]
box_blur_normalized = np.array([[x / 9 for x in row] for row in box_blur])
gaussian_3x3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]
gaussian_3x3_normalized = np.array([[x / 16 for x in row] for row in gaussian_3x3])
gaussian_5x5 = [
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
]
gaussian_5x5_normalized = np.array([[x / 256 for x in row] for row in gaussian_5x5])
print(gaussian_5x5_normalized)
output = cv2.filter2D(img, -1, ridge)
cv2.imshow('edge', output)
cv2.waitKey(0)
