import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('fashion_mnist_model.h5')  # Load the saved model
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0  # normalize the data

num = 10
predict = model.predict(f_image_test[:num])

print(" * Prediction: ", np.argmax(predict, axis=1))
print(" * Label: ", f_label_test[:num])
#
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1) #1st row, 4 columns, i+1 is the index of the subplot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_test[i])
    plt.xlabel(class_names[f_label_test[i]]) #idiot

plt.show()