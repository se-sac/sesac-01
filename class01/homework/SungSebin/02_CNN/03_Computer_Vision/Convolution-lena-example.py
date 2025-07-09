import cv2
import numpy as np
import matplotlib.pyplot as plt

# 흑백 이미지 불러오기
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# 엣지 검출 필터 (Laplacian-like)
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# 필터 적용
filtered_img = cv2.filter2D(img, -1, kernel)

# 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edge Detection')
plt.imshow(filtered_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()