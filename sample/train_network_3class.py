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
import pickle

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


# grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
def get_train_data():
    fi = open('img2level.pkl', 'rb')
    img2level = pickle.load(fi)
    fi.close()

    electronic_set = set(img2level['电子产品'])
    clothes_set = set(img2level['服饰与配饰'])
    other_set = set(
        list(img2level['init'] + img2level['家用电器'] + img2level['食品'] +
             img2level['母婴用品']))
    image_folders = ['shopline_1', 'shopline_3', 'shopline_4']

    imagePaths = []
    imageLabels = []
    for tmp_folder in image_folders:
        list_files = os.listdir(tmp_folder)
        for tmp_file in list_files:
            path_file = tmp_folder + '/' + tmp_file
            tmp_key = tmp_file.split('.')[0]

            try:
                Image.open(path_file)
                # imagePaths.append(path_file)
                if tmp_key in electronic_set:
                    imageLabels.append(0)
                    imagePaths.append(path_file)
                elif tmp_key in clothes_set:
                    imageLabels.append(1)
                    imagePaths.append(path_file)
                elif tmp_key in other_set:
                    imageLabels.append(2)
                    imagePaths.append(path_file)
            except:
                continue

    data = pd.read_csv('filter_data.csv')
    data_image = data['img_name']
    data_label = data['primary_category_id']
    select_total = 100  # 22048  #100
    select_num = 0
    for ind in range(len(data_label)):
        try:
            Image.open(data_image[ind])
            if data_label[ind] == 1:
                imageLabels.append(0)
                imagePaths.append(data_image[ind])
                select_num += 1
            if select_num > select_total:
                break
        except:
            continue

    return imagePaths, imageLabels


imagePaths, imageLabels = get_train_data()

random.seed(42)
# random.shuffle(imagePaths)
randomNum = np.random.permutation(len(imageLabels))

labels = []
data = []
# loop over the input images

for ind in randomNum:
    # load the image, pre-process it, and store it in the data list
    imagePath = imagePaths[ind]
    try:
        image = np.array(Image.open(imagePath).convert('L'), 'f') / 255.0
        # image = cv2.resize(image, (32, 32)) # accuracy: 0.8124
        image = cv2.resize(image, (64, 64))  # accuracy: 0.8355
        # data.append(image)
        # extract the class label from the image path and update the
        # labels list
        labels.append(imageLabels[ind])
        data.append(image)
    except:
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
                                                  test_size=0.20,
                                                  random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

trainX = trainX[:, :, :, np.newaxis]
testX = testX[:, :, :, np.newaxis]
# initialize the model
print("[INFO] compiling model...")
model = LeNet(trainX[0].shape, 3)
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

model.evaluate(testX, testY)
print('confusion matrix: ')

preds = model.predict(testX)
print("preds: ",
      np.argmax(testY, axis=1).shape,
      np.argmax(preds, axis=1).shape)
print(
    tf.math.confusion_matrix(np.argmax(testY, axis=1),
                             np.argmax(preds, axis=1),
                             num_classes=3))
