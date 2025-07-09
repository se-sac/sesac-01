import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('mnist_model.h5')
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

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