# TensorFlow and tf.keras
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

os.environ["PATH"] += os.pathsep + 'D:\\软件\\graphviz\\bin'

# Helper libraries
import numpy as np


def change_shape(images):
    # Change the shape to (48, 48, 3)
    images = np.reshape(images, (len(images), 28, 28, 1))
    # Current shape (len, 28, 28, 1)
    images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images))
    # Current shape (len, 28, 28, 3)
    images = np.array(tf.image.resize(images, [71, 71]))
    # Current shape (48, 48, 3)
    # Normalise the data and change data type
    images = images / 255.
    images = images.astype('float32')
    # Preprocess input
    return images


# load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# resize the img
train_images = change_shape(train_images)
test_images = change_shape(test_images)

# input layer
input_img = keras.layers.Input(shape=(71, 71, 3))

# pretrained model
conv_base = keras.applications.Xception(weights='imagenet',
                                        include_top=False,
                                        input_shape=(71, 71, 3))

# Freeze the layers except the last 4 layers
for layer in conv_base.layers[:-4]:
    layer.trainable = False

# custom layer
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# plot
keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# compile
model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=20, batch_size=512, validation_split=0.2)

# test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
