import tensorflow as tf
from tensorflow import keras
# Helper libraires
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def prepare(ds, batch = 1,  shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # Resize and rescale all datasets.
    # x: image, y: label
    
    # 이미지 크기 조정
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
                num_parallel_calls=AUTOTUNE)
          
    # Batch all datasets
    ds = ds.batch(batch_size)
    
    # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해 
    # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미리 로드 하도록 함.
    return ds.prefetch(buffer_size=AUTOTUNE)

# Grad CAM hitmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # MobileNetV3Small 모델 가져오기
    mobilenet = model.get_layer('MobileNetV3Small')
    
    # 마지막 conv 레이어와 모델의 출력 가져오기
    last_conv_layer = mobilenet.get_layer(last_conv_layer_name)
    last_conv_output = last_conv_layer.output

    # 새로운 모델 생성 (마지막 conv layer -> 원래 모델의 끝)
    grad_model = tf.keras.Model(
        inputs=[mobilenet.inputs],
        outputs=[last_conv_output, mobilenet.output]
    )

    print("img_array.shape", img_array.shape)
    # 입력 이미지에 대한 그래디언트 계산
    with tf.GradientTape() as tape:
        last_conv_output, predictions = grad_model(img_array)
        
        class_channel = predictions[:, tf.argmax(predictions[0])]
        
    # 출력에 대한 마지막 conv layer의 그래디언트 계산
    grads = tape.gradient(class_channel, last_conv_output)
    
    # 각 필터의 중요도를 나타내는 가중치 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 가중치와 마지막 conv layer의 출력을 곱하여 채널별 히트맵 생성
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]
        
    # 채널 평균을 취해 히트맵 생성
    heatmap = np.mean(last_conv_output, axis=-1)
    
    # 히트맵 정규화
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def save_and_display_gradcam(img, idx, idx_name, model, last_conv_layer_name, alpha=0.4):
    # 원본 이미지용 배열 준비
    img_array = img
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # GradCAM 히트맵 생성
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # 히트맵을 0-255 범위로 변환
    heatmap = np.uint8(255 * heatmap)

    # jet colormap 사용하여 히트맵 컬러화
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # 히트맵 크기를 원본 이미지 크기로 조정
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # 히트맵과 원본 이미지 합성
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # 결과 저장
    cam_path = f"cam_{idx}_{idx_name}.jpg"
    superimposed_img.save(cam_path)

    
img_height  = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 알아서 처리하도록.

# Dataset 준비 (https://www.tensorflow.org/tutorials/load_data/images?hl=ko)
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num = 20
num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes, ", type: ", type(label_name))

test_ds = prepare(test_ds, num)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype='int')

# 모델 불러오기
MobileNetV3Small = tf.keras.applications.MobileNetV3Small(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape = (img_height, img_width, 3),
    include_top = False)

MobileNetV3Small.summary()
model = tf.keras.models.load_model('transfer_learning_flower.keras')
model.summary()

predict = model.predict(image_test)
predicted_classes = np.argmax(predict, axis=1)

# 실제 레이블과 예측 출력
print("실제 레이블 | 예측 레이블");
print("------------------------")
for ll in range((label_test.size)):
    print(label_name[label_test[ll]], "|", label_name[predicted_classes[ll]])
print("------------------------")    
# print("실제 레이블:", [label_name[idx] for idx in label_test])
# print("예측 레이블:", [label_name[idx] for idx in predicted_classes])

# 정확도 계산도 추가할 수 있습니다
accuracy = np.mean(predicted_classes == label_test)
print(f"정확도: {accuracy:.2%}")

# Grad CAM
# mobilenet = model.get_layer('MobileNetV3Small')
# last_conv_layer = None
# for layer in mobilenet.layers:
#     print(layer)
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         last_conv_layer = f'MobileNetV3Small/{layer.name}'

last_conv_layer_name = 'conv_1'

# 각 이미지에 대해 GradCAM 생성 및 표시
for i in range(min(5, len(image_test))):  # 처음 5개 이미지만 표시
    print(f"\nImage {i+1}")
    print(f"Actual: {label_name[label_test[i]]}")
    print(f"Predicted: {label_name[predicted_classes[i]]}")
    save_and_display_gradcam(
        image_test[i],
        i,
        f"{label_name[label_test[i]]}_{label_name[predicted_classes[i]]}",
        model,
        last_conv_layer_name
    )