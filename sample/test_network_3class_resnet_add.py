from keras.applications.resnet50 import ResNet50
from tensorflow import keras
from keras.applications.resnet50 import ResNet50
import os
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow.keras.applications.resnet_v2 as resnetv2
from sklearn.metrics import classification_report
from PIL import Image
import datetime
from matplotlib import pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def get_pretrained_resnet(num_classes=3):
    resnet50_fine_tune = keras.models.Sequential()
    resnet50_fine_tune.add(
        resnetv2.ResNet101V2(include_top=False,
                             pooling='avg',
                             weights='imagenet'))
    resnet50_fine_tune.add(
        keras.layers.Dense(num_classes, activation='softmax'))
    # resnet50_fine_tune.layers[0].trainable = False

    resnet50_fine_tune.compile(loss="categorical_crossentropy",
                               optimizer="sgd",
                               metrics=["accuracy"])
    resnet50_fine_tune.summary()

    return resnet50_fine_tune


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
                img = Image.open(path_file)
                del img
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
            img = Image.open(data_image[ind])
            del img
            if data_label[ind] == 1:
                imageLabels.append(0)
                imagePaths.append(data_image[ind])
                select_num += 1
            if select_num > select_total:
                break
        except:
            continue

    return imagePaths, imageLabels


def get_test_data(input_data_csv, select_num=500):
    # select test_data from `input_data_csv`
    test_csv = pd.read_csv(input_data_csv)
    drop_dup_csv = test_csv.drop_duplicates(subset=['img_name'], keep=False)
    testImagePaths = []
    testImageLabels = []
    csv_label = drop_dup_csv['primary_category_id']
    csv_path = drop_dup_csv['img_name']

    print("csv_summary; ", Counter(csv_label))

    start_num1 = 0
    start_num2 = 0
    start_num3 = 0
    for ind in range(len(csv_label)):
        try:
            img = Image.open(csv_path[ind])
            del img
            if int(csv_label[ind]) == 77:
                if ind > 20000:
                    # 如果是衣服类别，标签设置为1
                    testImageLabels.append(1)
                    testImagePaths.append(csv_path[ind])
                    start_num1 += 1
            elif int(csv_label[ind]) == 1:
                # 如果是3C类别，标签设置为0
                if start_num2 < select_num:
                    testImageLabels.append(0)
                    testImagePaths.append(csv_path[ind])
                start_num2 += 1
            else:
                testImageLabels.append(2)
                testImagePaths.append(csv_path[ind])
                start_num3 += 1

            if start_num1 > select_num and start_num2 > select_num:
                break
        except:
            continue

    print("test_num: ", len(testImagePaths), len(testImageLabels))

    test_dict = {'filename': testImagePaths, 'class': testImageLabels}
    test_df = pd.DataFrame(test_dict, dtype=str)

    # 将测试数据处理成resnet50模型的接入数据
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.resnet50.preprocess_input)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=24,
        seed=7,
        shuffle=False,
        # label的编码格式：这里为one-hot编码
        class_mode='categorical')

    test_generator.reset()
    return test_generator, len(testImagePaths)


def get_error_list(error_file_path, select_num=2000):
    pass


def plot_learning_curves(history,
                         label,
                         epochs,
                         min_value,
                         max_value,
                         name="accuracy"):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.savefig(name + ".png")


if __name__ == '__main__':
    input_csv = '/data1/seven/workspace/models/cloth/data/train_1225.csv'
    test_datagen, total = get_test_data(input_csv)
    # test_datagen.filenames
    # test_datagen.next()[0][1]
    # print(test_datagen.)

    model_weight_path = '/data1/xusong/shopline_data/train_model_3class_resnet_2w_add.h5'
    resnet50_fine_tune = get_pretrained_resnet()

    # checkpoint_path = 'logs/resnet_cp.ckpt.data-00000-of-00001'
    # resnet50_fine_tune.load_weights(checkpoint_path)
    resnet50_fine_tune.load_weights(model_weight_path)

    print("original test labels: ", len(test_datagen.classes))

    predIdxs = resnet50_fine_tune.predict_generator(test_datagen)
    predict_labels = np.argmax(predIdxs, axis=1)
    print(
        tf.math.confusion_matrix(test_datagen.classes,
                                 predict_labels,
                                 num_classes=3))
    # print(classification_report(test_datagen.classes, predict_labels))
    print("error_id: ", test_datagen.classes != predict_labels)
