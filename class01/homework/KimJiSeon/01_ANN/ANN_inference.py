import tensorflow as tf
# Helper libraires
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('mnist_ANN_Batch.keras')
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))