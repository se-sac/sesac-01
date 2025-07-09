import numpy as np  # for linear algebra
import matplotlib.pyplot as plt  # for plotting things
import os
from PIL import Image  # for reading images

# keras Libraries < - CNN
#import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # < - define evaluation metrics

mainDIR = './chest_xray/'

# train / test / val
train_folder = mainDIR + 'train/'
test_folder  = mainDIR + 'test/'
val_folder   = mainDIR + 'val/'

# train/
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

#Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('#Normal pics:', norm_pic)
norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('Pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# let plt these images
f = plt.figure(figsize=(10,6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

#ANN model
model_in = Input(shape=(64, 64, 3))

x = Flatten()(model_in)
x = Dense(activation='relu', units=256)(x)
x = Dense(activation='relu', units=128)(x)
x = Dense(activation='relu', units=64)(x)
x = Dense(activation='sigmoid', units=1)(x)

model_fin = Model(inputs=model_in, outputs=x)

model_fin.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# # let's build the CNN model
# model = Sequential()

# # 1st Convolutional Layer
# model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # 2nd Convolutional Layer
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # 3rd Convolutional Layer
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Flatten + Dense
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

# # Compile the CNN
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # 모델 구조 확인
# model.summary()


num_of_test_samples = 600
batch_size = 32

# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)  # image normalization

training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Validation set generator
validation_generator = test_datagen.flow_from_directory('./chest_xray/val',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')

# Test set generator
test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model_fin.summary()



history = model_fin.fit(
    training_set,
    steps_per_epoch=len(training_set),  # 전체 학습 샘플 수 / batch_size
    epochs=10,                          # 원하는 epoch 수
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()

model_fin.save('pneumonia_model.h5')