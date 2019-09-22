# based on code from https://www.tensorflow.org/tutorials

import tensorflow as tf
import numpy as np
import cv2

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# specify path to training data and testing data

folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + "x_train.csv"
train_y_location = foldersmall + "/" + "y_train.csv"
test_x_location = folderbig + "/" + "x_test.csv"
test_y_location = folderbig + "/" + "y_test.csv"

print("Reading training data")
x_train_2d = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
x_train_3d = x_train_2d.reshape(-1,28,28,1)
x_train = x_train_3d
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

print("Pre processing x of training data")
x_train = x_train / 255.0

# erosion: preprocessing to improve the visibility to the model
for image_index in range(x_train.shape[0]):
  kernel = np.ones((2,2),np.uint8)
  res = cv2.erode(x_train[image_index,:,:,:],kernel,iterations = 1)
  res = res.reshape(-1,28,28,1)
  x_train[image_index,:,:,:] = res


# Residual network
inputs = tf.keras.Input(shape=(28,28,1))
intermediate_output = tf.keras.layers.MaxPool2D(4, 4, padding='same')(inputs)
intermediate_output = tf.keras.layers.Conv2D(128, (2,2), padding='same')(intermediate_output)
intermediate_output = tf.layers.batch_normalization(intermediate_output)
intermediate_output = tf.keras.layers.Activation("relu")(intermediate_output)
intermediate_output = tf.keras.layers.Conv2D(256, (2,2), padding='same')(intermediate_output)
intermediate_output = tf.layers.batch_normalization(intermediate_output)
intermediate_output = tf.keras.layers.Activation("relu")(intermediate_output)
block_1_output = intermediate_output

intermediate_output = tf.keras.layers.Conv2D(128, 2, padding='same')(block_1_output)
intermediate_output = tf.layers.batch_normalization(intermediate_output)
intermediate_output = tf.keras.layers.Activation("relu")(intermediate_output)
intermediate_output = tf.keras.layers.Conv2D(256, 2, padding='same')(intermediate_output)
intermediate_output = tf.layers.batch_normalization(intermediate_output)
intermediate_output = tf.keras.layers.Activation("relu")(intermediate_output)
block_2_output = tf.keras.layers.add([intermediate_output, block_1_output])

intermediate_output = tf.keras.layers.Conv2D(256, 2, padding='same')(block_2_output)
intermediate_output = tf.layers.batch_normalization(intermediate_output)
intermediate_output = tf.keras.layers.Activation("relu")(intermediate_output)
intermediate_output = tf.keras.layers.Flatten()(intermediate_output)
intermediate_output = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(intermediate_output)
intermediate_output = tf.keras.layers.Dropout(0.1)(intermediate_output)
outputs = tf.keras.layers.Dense(10, activation='softmax')(intermediate_output)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=10)



print("Reading testing data")
x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
x_test_3d = x_test_2d.reshape(-1,28,28,1)
x_test = x_test_3d
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

print("Pre processing testing data")
x_test = x_test / 255.0

# erosion
for image_index in range(x_test.shape[0]):
  kernel = np.ones((2,2),np.uint8)
  res = cv2.erode(x_test[image_index,:,:,:],kernel,iterations = 1)
  res = res.reshape(-1,28,28,1)
  x_test[image_index,:,:,:] = res


print("evaluate")
model.evaluate(x_test, y_test)
