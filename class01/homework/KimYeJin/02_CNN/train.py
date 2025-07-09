import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train),(f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0 #normalize the data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1) #1st row, 4 columns, i+1 is the index of the subplot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])

plt.show()

#CNN -> ANN 위에 CNN을 쌓는 구조
model = tf.keras.Sequential() #model is a linear stack of layers
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) #input_shape is the shape of the input data
model.add(tf.keras.layers.MaxPooling2D((2, 2))) #pooling layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #another convolutional layer
model.add(tf.keras.layers.MaxPooling2D((2, 2))) #another pooling layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #another convolutional layer

#ANN
model.add(tf.keras.layers.Flatten()) #flatten the output of the convolutional layers to feed
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #sparse -> 각 이미지에 대한 클래스값
              metrics=['accuracy'])
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('fashion_mnist.h5') #.h5는 keras 모델 타입 .h5 말고 .keras도 가능