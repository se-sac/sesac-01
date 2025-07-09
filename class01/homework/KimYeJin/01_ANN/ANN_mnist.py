import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train),(f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0 #normalize the data

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1) #1st row, 4 columns, i+1 is the index of the subplot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(f_label_train[i])

plt.show()

model = tf.keras.Sequential() #model is a linear stack of layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #sparse -> 각 이미지에 대한 클래스값
              metrics=['accuracy'])
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist_model.h5') #.h5는 keras 모델 타입 .h5 말고 .keras도 가능
                                                                                    

                          
                                                                    