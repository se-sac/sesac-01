import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
print("Train Image shape : ", image_train.shape)
print("Train Label : ", label_train, "\n")
print(image_train[0])
num = 10
for idx in range (num):
    sp = plt.subplot(5,5,idx+1)
    plt.imshow(image_train[idx])
    plt.title(f'Label: {label_train[idx]}')
plt.show()

model = Sequential()
model.add(Flatten())
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(64,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))

model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
history = model.fit(image_train, label_train, validation_data=(image_test,label_test), epochs=10)
model.summary()
model.save('mnist_ANN_Batch.keras')

loss = history.history["loss"]
acc = history.history["accuracy"]
val_loss = history.history["val_loss"]
val_acc  = history.history["val_accuracy"]
plt.subplot(1,2,1)
plt.plot(range(len(loss)),loss,label = "Train Loss")
plt.plot(range(len(val_loss)),val_loss,label = "Validation Loss")
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(acc)),acc,label = "Train Accuracy")
plt.plot(range(len(val_acc)),val_acc,label = "Validation Accuracy")
plt.grid()
plt.legend()
plt.show()
