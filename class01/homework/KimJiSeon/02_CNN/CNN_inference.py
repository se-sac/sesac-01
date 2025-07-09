import tensorflow as tf
# Helper libraires
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('CNN_mnist.keras')
mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

num = 20
predict = model.predict(f_image_test[:num])
print(f_label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))

 