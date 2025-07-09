# import tensorflow
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# 이미지 전처리를 위한 설정
image_height = 255
image_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

# Dataset 준비
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)
num = 20
# 이미지 전처리 함수 정의
def prepare_ds(ds, batch=1, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # 이미지 크기 조정 및 전처리
    ds = ds.map(lambda x, y: (tf.image.resize(x, [image_height, image_width]), y),
                num_parallel_calls=AUTOTUNE)

    # MobileNetV2용 전처리 적용
    ds = ds.map(lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)

    # 배치 구성
    ds = ds.batch(batch_size)

    # 성능 최적화를 위해 prefetch 적용 (다음 배치 미리 로드)
    return ds.prefetch(buffer_size=AUTOTUNE)

# 클래스 정보 출력
num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print("label_name:", label_name, ", classes:", num_classes, ", type:", type(label_name))

# 테스트셋 준비
test_ds = prepare_ds(test_ds)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype=int)

# 저장된 모델 불러오기
model = tf.keras.models.load_model('transfer_learing_flower.keras')

# 모델 구조 요약
model.summary()

# 추론 수행
predict = model.predict(image_test)
predicted_classes = np.argmax(predict, axis=1)

# 실제 레이블 vs 예측 레이블 출력
print("실제 레이블 → 예측 레이블:")
for i in range(len(label_test)):
    print(f"{label_name[label_test[i]]} → {label_name[predicted_classes[i]]}")

# 정확도 계산
accuracy = np.mean(predicted_classes == label_test)
print(f"정확도: {accuracy * 100:.2f}%")
