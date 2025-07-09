import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import numpy as np
import os

img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

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

with open("history_pneumonia", "rb") as pf:
    history = pickle.load(pf)
# Accuracy ​
plt.figure()
plt.subplot(1,2,1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train set', 'Validation set'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(history['val_loss'])
plt.plot(history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train set', 'Validation set'], loc='upper left')
plt.savefig('training_accuracy_loss.png')
plt.show(block=False)

model = tf.keras.models.load_model('transfer_learning_pneumonia.keras')
model.summary()

path = '../99_vision_pneumonia/'
train_path = 'chest_xray/train/'
test_path = 'chest_xray/test/'
val_path = 'chest_xray/val/'

test_n = os.path.join(path, test_path, 'NORMAL/')
test_p = os.path.join(path, test_path, 'PNEUMONIA/')

img_list_n = os.listdir(test_n)
img_list_p = os.listdir(test_p)

test_ds = image_dataset_from_directory(os.path.join(path, test_path),
                                        validation_split=0.2,
                                        subset="validation",
                                        seed=123,
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size)

test_ds = prepare(test_ds)
test_img, test_label = next(iter(test_ds))
test_img = np.array(test_img)
test_label = np.array(test_label)
# print(test_img.shape, test_label.shape)
predict = model(test_img)
predict = np.array(predict)
Class_name = ['NORMAL', 'PNEUMONIA'] 
print('NORMAL', ' | ', 'PNEUMONIA')
plt.figure(figsize=(28,28))
test_img = test_img/255
for idx in range(25):
    target_idx = np.argmax(test_label[idx])
    predict_idx = np.argmax(predict[idx]).astype(int).squeeze()
    print(Class_name[target_idx], " | ", Class_name[predict_idx])
    plt.subplot(5,5,idx+1)
    plt.imshow(test_img[idx:idx+1].squeeze())
    plt.axis('off')
    plt.title(f'Label target: {Class_name[target_idx]}' + '\n' + 
              f'Label predict: {Class_name[predict_idx]}' , fontdict= {'fontsize': 22})
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig('test_results.jpg')
plt.show()