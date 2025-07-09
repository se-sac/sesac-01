import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 작업 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 저장된 모델 로드
model = load_model('medical_ann.h5')
print(model.summary())  # 모델 구조 확인

# 테스트 데이터 제너레이터: 정규화만 수행
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    './chest_xray/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 테스트 세트 전체 인덱스
num_samples = test_set.samples
indices = np.random.choice(num_samples, size=25, replace=False)

# 배치별 이미지와 라벨, 예측 확률 획득
images = []
true_labels = []
pred_labels = []

for idx in indices:
    # batch index, within-batch offset 계산
    batch_idx = idx // test_set.batch_size
    batch_offset = idx % test_set.batch_size

    # 해당 배치 로드
    batch_imgs, batch_lbls = test_set.next() if batch_idx == 0 else test_set[batch_idx]
    img = batch_imgs[batch_offset]
    lbl = batch_lbls[batch_offset]

    # 모델 예측
    prob = model.predict(img[np.newaxis, ...])[0][0]
    pred = 1 if prob > 0.5 else 0

    images.append(img)
    true_labels.append(lbl)
    pred_labels.append(pred)

plt.figure(figsize=(15, 15))
for i, (img, true, pred) in enumerate(zip(images, true_labels, pred_labels)):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(img)
    color = 'green' if true == pred else 'red'
    plt.title(f'True: {int(true)}, Pred: {pred}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()


