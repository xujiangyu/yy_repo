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