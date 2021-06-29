from keras import activations
from keras.backend import flatten
from keras.layers.pooling import MaxPool2D
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import cifar10
import glob
import string
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_val, y_val = x_train[40000:50000], y_train[40000:50000]
x_train, y_train = x_train[:40000], y_train[:40000]
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_val = x_val.reshape(x_val.shape[0], 32, 32, 3)
x_test = x_train.reshape(x_train.shape[0], 32, 32, 3)
# scale pixels
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
# Standardizing (255 is the total number of pixels an image can have)
x_train = x_train / 255
x_val = x_val / 255
x_test = x_test / 255

# One hot encoding the target class (labels)
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding="same",
                 kernel_initializer="he_uniform", input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(Conv2D(256, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(Conv2D(256, (3, 3), activation='relu',
                 padding="same", kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(units=256, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

numOfEpoch = 10
with tf.device('/device:GPU:0'):
    H = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  batch_size=100, epochs=10, verbose=1)

plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch),
         H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch),
         H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()
model.save('model.ckpt')
