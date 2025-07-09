import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D
from tensorflow.keras.utils import image_dataset_from_directory
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

img_height  = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 알아서 처리하도록.

# Dataset 준비
path = '../99_vision_pneumonia/'
train_path = 'chest_xray/train/'
test_path = 'chest_xray/val/'
val_path = 'chest_xray/test/'

train_n = os.path.join(path, train_path, 'NORMAL/')
train_p = os.path.join(path, train_path, 'PNEUMONIA/')

img_list_n = os.listdir(train_n)
img_list_p = os.listdir(train_p)

train_ds = image_dataset_from_directory(os.path.join(path, train_path),
                                        validation_split=0.2,
                                        subset="training",
                                        seed=123,
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size)
val_ds = image_dataset_from_directory(os.path.join(path, val_path),
                                        validation_split=0.2,
                                        subset="validation",
                                        seed=123,
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size)
# image_train, label_train = next(iter(train_ds))
# image_val , label_val  = next(iter(val_ds))
# image_test, label_test = next(iter(test_ds))

num_classes = 2
label_name = ['NORMAL', 'PNEUMONIA'] 
print(label_name, ", classnum : ", num_classes)

def prepare(ds, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # Resize and rescale all datasets.
    # x: image, y: label
    
    # 이미지 크기 조정
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
                num_parallel_calls=AUTOTUNE)
    
    # 전처리 적용
    ds = ds.map(lambda x, y: (preprocess_input(x), y), 
                num_parallel_calls=AUTOTUNE)
        
    # # Batch all datasets
    # ds = ds.batch(batch_size)
    
    # Use data augmentation only on the training set.
    if augment:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE)
        
        
    # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해 
    # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미리 로드 하도록 함.
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)


# https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small
# include_top -> ANN 부분 직접 수정
base_model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape = (img_height, img_width, 3),
    include_top = False)
# 기본 모델의 가중치 동결
base_model.trainable = False

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./256, offset=-1)

# https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
# 추론, 학습에서 다르게 동작하는 layer들을 추론/학습 중 하나로만 동작하게 함. 
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer = 'adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(train_ds, epochs=15, validation_data=val_ds)


model.save('transfer_learning_pneumonia.keras')
with open('history_pneumonia', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


