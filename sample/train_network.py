# set the matplotlib backend so figures can be saved in the background
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import pandas as pd
#import argparse
import random
import cv2
import os
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# LeNet-5 model
# LeNet-5 model
class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(
            Conv2D(6,
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   activation='tanh',
                   input_shape=input_shape,
                   padding="same"))
        self.add(
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'))
        self.add(
            Conv2D(16,
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   activation='tanh',
                   padding='valid'))
        self.add(
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='tanh'))
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(nb_classes, activation='softmax'))

        self.compile(optimizer='adam',
                     loss=categorical_crossentropy,
                     metrics=['accuracy'])


# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-m", "--model", required=True, help="path to output model")
# ap.add_argument("-p",
#                 "--plot",
#                 type=str,
#                 default="plot.png",
#                 help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
imageCsv = pd.read_csv('filter_data.csv')
imagePaths = imageCsv['img_name']
imageLabels = imageCsv['primary_category_id']

random.seed(42)
# random.shuffle(imagePaths)
randomNum = np.random.permutation(len(imageLabels))

# loop over the input images
total1 = 0
total2 = 0
total_sample = 100000
for ind in randomNum:
    # load the image, pre-process it, and store it in the data list
    imagePath = imagePaths[ind]
    try:
        image = np.array(Image.open(imagePath).convert('L'), 'f') / 255.0
        image = cv2.resize(image, (28, 28))
        # data.append(image)
        # extract the class label from the image path and update the
        # labels list
        imageLabel = imageLabels[ind]
        print('image_shape: ', image.shape)
        #print("imageLabel", type(imageLabel))
        if imageLabel == 1:
            label = 1
            #total1 += 1
            if total1 < total_sample:
                total1 += 1
                data.append(image)
                labels.append(label)
        else:
            label = 0
            #total2 += 1
            if total2 < total_sample:
                total2 += 1
                data.append(image)
                labels.append(label)
        print('total1 :', total1)
        print('total2 :', total2)
        if total1 >= total_sample and total2 >= total_sample:
            break
        # labels.append(label)
    except:
        cmd = 'cp ' + imagePath + ' ' + './error_images'
        os.popen(cmd)
        continue
# =============================================================================
#         image = cv2.imread(imagePath, 0)
#         image = cv2.resize(image, (28, 28))
#         image = img_to_array(image)
# =============================================================================

# scale the raw pixel intensities to the range [0, 1]
print('data len', len(data))
data = np.array(data)
labels = np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

trainX = trainX[:, :, :, np.newaxis]
testX = testX[:, :, :, np.newaxis]
# initialize the model
print("[INFO] compiling model...")
model = LeNet(trainX[0].shape, 2)
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# # trainY = np.array(trainY)
# # trainX = np.array(trainX)
# print("shape: ", trainX.shape, trainY.shape)
# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=60,
#                          width_shift_range=0.1,
#                          height_shift_range=0.1,
#                          shear_range=0.2,
#                          zoom_range=0.2,
#                          horizontal_flip=True,
#                          fill_mode="nearest")

# train the network
print("[INFO] training network...")
# Add a new axis

log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Specify the callback object
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

# train_genetor = aug.flow(trainX, trainY, batch_size=BS)
model.fit(trainX,
          y=trainY,
          epochs=20,
          validation_data=(testX, testY),
          callbacks=[tensorboard_callback],
          verbose=0)

# save the model to disk
print("[INFO] serializing network...")
model.save('train_model.h5')
#preds = model.predict(testX)
model.evaluate(testX, testY)
