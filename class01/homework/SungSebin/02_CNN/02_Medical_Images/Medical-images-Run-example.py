import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# 모델 로드
model = tf.keras.models.load_model('./pneumonia_model.h5')

# 테스트 이미지 폴더 (NORMAL 또는 PNEUMONIA 중 하나)
test_dir = './chest_xray/test/NORMAL/'  # 또는 'PNEUMONIA/'
img_files = os.listdir(test_dir)
num_images = 25  # 5x5

plt.figure(figsize=(15, 15))

for i in range(num_images):
    img_path = os.path.join(test_dir, img_files[i])
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred

    # 시각화
    plt.subplot(5, 5, i + 1)
    plt.imshow(img)
    plt.title(f"{label}\nConf: {confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()