# TensorFlow and tf.keras
import os

import tensorflow as tf
from tensorflow import keras

os.environ["PATH"] += os.pathsep + 'D:\\software\\graphviz\\bin'

# Helper libraries

# load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale values to range 0 to 1
train_images = train_images / 255.0
train_images = train_images.reshape(60000, 28, 28, -1)
test_images = test_images / 255.0
test_images = test_images.reshape(10000, 28, 28, -1)

# model input
input_img = keras.layers.Input(shape=(28, 28, 1))

# conv layer
layer_1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
layer_1 = keras.layers.MaxPooling2D((2, 2))(layer_1)

layer_2 = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(input_img)
layer_2 = keras.layers.MaxPooling2D((2, 2))(layer_2)

layer_3 = keras.layers.Conv2D(32, (7, 7), padding='same', activation='relu')(input_img)
layer_3 = keras.layers.MaxPooling2D((2, 2))(layer_3)

# concatenate output of the former layers
mid_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis=3)

# flatten
flat_1 = keras.layers.Flatten()(mid_1)

# dense layer
dense_1 = keras.layers.Dense(64, activation='relu')(flat_1)
batch_1 = keras.layers.BatchNormalization()(dense_1)
drop_1 = keras.layers.Dropout(rate=0.5)(batch_1)

# dense layer
dense_2 = keras.layers.Dense(64, activation='relu')(drop_1)
batch_2 = keras.layers.BatchNormalization()(dense_2)
drop_2 = keras.layers.Dropout(rate=0.5)(batch_2)

# output
output = keras.layers.Dense(10, activation='softmax')(drop_2)

# build model
model = keras.Model([input_img], output)

# plot
keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# compile
model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
history = model.fit(train_images, train_labels, epochs=20, batch_size=256, validation_split=0.2)

# test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
