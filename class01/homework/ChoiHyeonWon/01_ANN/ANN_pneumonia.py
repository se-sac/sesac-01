import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import os
import pickle

img_height = 64
img_width = 64
batch_size = 32

# kaggle data load (pneumonia)
path = '../99_vision_pneumonia/'
train_path = 'chest_xray/train/'
test_path = 'chest_xray/test/'
val_path = 'chest_xray/val/'

train_n = os.path.join(path, train_path, 'NORMAL/')
train_p = os.path.join(path, train_path, 'PNEUMONIA/')

img_list_n = os.listdir(train_n)
img_list_p = os.listdir(train_p)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

train_ds = train_datagen.flow_from_directory(os.path.join(path, train_path),
                                                target_size = (img_height, img_width),
                                                batch_size = batch_size,
                                                class_mode = 'binary')

val_ds = val_datagen.flow_from_directory(os.path.join(path, val_path),
                                                target_size = (img_height, img_width),
                                                batch_size = batch_size,
                                                class_mode='binary')

model_in = tf.keras.Input(shape=(img_height, img_width, 3))
x = tf.keras.layers.Flatten()(model_in)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
model_out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
model = tf.keras.Model(inputs=model_in, outputs=model_out)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(), 
            metrics=['accuracy'])

history = model.fit(train_ds,
          validation_data = val_ds,
          epochs = 10)


model.save('penumonia.keras')

with open('history_xray','wb') as pf:
    pickle.dump(history, pf)

