#normal vs pneumonia classification
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 스크립트 파일이 있는 디렉토리로 작업 디렉토리 변경
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

mainDIR = os.listdir('./chest_xray')
print(mainDIR)

train_folder = './chest_xray/train'
val_folder = './chest_xray/val'
test_folder = './chest_xray/test'

#train
os.listdir(train_folder)
train_n = os.path.join(train_folder, 'NORMAL')
train_p = os.path.join(train_folder, 'PNEUMONIA')

#Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title:', norm_pic)
norm_pic_address = os.path.join(train_n, norm_pic)

#Pneumonia pic
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = os.path.join(train_p, sic_pic)
print('pneumonia picture title:', sic_pic)

#Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plot these images
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()


#let's build ANN model

num_of_test_samples = 600
batch_size = 32

#Fitting the ANN to the images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255) #Image normalization

training_set= train_datagen.flow_from_directory('./chest_xray/train',
                                               target_size=(64, 64),
                                               batch_size=batch_size,
                                               class_mode='binary')

validation_set = test_datagen.flow_from_directory('./chest_xray/val',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size=(64, 64),
                                            batch_size= 32,
                                            class_mode='binary')

model_in = Input(shape=(64, 64, 3)) #input shape is 64x64x3
model = Flatten()(model_in) #flatten the input
#Fully connected layers
model = Dense(activation='relu', units = 128)(model) #first hidden layer
model = Dense(activation='sigmoid', units = 1)(model) #final output layer

#Compile the Neural Network
model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model_fin.summary()


ann_model = model_fin.fit(
    training_set,  
    steps_per_epoch=163,# 학습 데이터 제너레이터
    epochs=10,                     # 원하는 에폭 수
    validation_data=validation_set,
    validation_steps = 624)

test_accu = model_fin.evaluate(test_set,steps=624)

model_fin.save('medical_ann.h5')  # Save the model
print('The testing accuracy is: ', test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, steps=624)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)


model = tf.keras.models.load_model('medical_ann.h5')
medical_ann = tf.keras.datasets.medical_ann
(f_image_train, f_label_train), (f_image_test, f_label_test) = medical_ann.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0  # normalize the data

num = 10
predict = model.predict(f_image_test[:num])

print(" * Prediction: ", np.argmax(predict, axis=1))
print(" * Label: ", f_label_test[:num])
#

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1) #1st row, 4 columns, i+1 is the index of the subplot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_test[i])
    plt.xlabel(f_label_test[i])

plt.show()