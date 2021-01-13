from tensorflow import keras
from keras.applications.resnet50 import ResNet50
import os
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.applications.resnet_v2 as resnetv2
from PIL import Image
import datetime
import random
from collections import Counter
from matplotlib import pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_pretrained_resnet(num_classes=3):
    resnet50_fine_tune = keras.models.Sequential()
    resnet50_fine_tune.add(
        resnetv2.ResNet50V2(include_top=False,
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


def get_balanced_train_data(shopline_data, xuanpin_data, total=100000):
    def check_image(img_name):
        try:
            img = Image.open(tmp_name)
            del img
            return True
        except:
            return False

    def extend_large_size(short_list, larg_size):
        random.seed(2021)
        random.shuffle(short_list)
        # large_list = None
        a = larg_size
        b = len(short_list)
        if b > a:
            return short_list[:a]
        div = a // b
        mod = a % b

        large_list = short_list * div + short_list[:mod]
        return large_list

    shopline_da = pd.read_csv(shopline_data)
    xuanpin_da = pd.read_csv(xuanpin_data)

    shopline_da_img_name = shopline_da['img_name']
    shopline_da_img_label = shopline_da['一级']

    xuanpin_da_img_name = xuanpin_da['img_name']
    xuanpin_da_img_label = xuanpin_da['primary_category_id']

    three_c_image_paths = []
    clothes_image_paths = []
    others_image_paths = []

    for ind1 in range(len(shopline_da_img_name)):
        tmp_name = shopline_da_img_name[ind1]
        tmp_label = shopline_da_img_label[ind1]
        if check_image(tmp_name):
            if tmp_label in ['init', '家用电器', '食品', '母婴用品']:
                others_image_paths.append(tmp_name)
            if tmp_label == '服饰与配饰':
                clothes_image_paths.append(tmp_name)
            if tmp_label == '电子产品':
                three_c_image_paths.append(tmp_name)

    for ind2 in range(len(xuanpin_da_img_name)):
        if check_image(tmp_name):
            tmp_name = xuanpin_da_img_name[ind2]
            tmp_label = xuanpin_da_img_label[ind2]
            if int(tmp_label) == 1:
                three_c_image_paths.append(tmp_name)
            else:
                clothes_image_paths.append(tmp_name)

    extend_clothes_paths = extend_large_size(clothes_image_paths, total)
    extend_threeC_paths = extend_large_size(three_c_image_paths, total)
    extend_others_paths = extend_large_size(others_image_paths, total)

    print(
        f'clothes original len: {len(clothes_image_paths)}, after extend len: {len(extend_clothes_paths)}'
    )

    print(
        f'3c original len: {len(three_c_image_paths)}, after extend len: {len(extend_threeC_paths)}'
    )

    print(
        f'others original len: {len(others_image_paths)}, after extend len: {len(extend_others_paths)}'
    )

    imagePaths = extend_threeC_paths + extend_clothes_paths + extend_others_paths
    imageLabels = len(extend_threeC_paths) * [0] + len(
        extend_clothes_paths) * [1] + len(extend_others_paths) * [2]

    return imagePaths, imageLabels


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
    select_total = 22048  #100
    select_num1 = 0
    # select_num2 = 0
    for ind in range(len(data_label)):
        try:
            img = Image.open(data_image[ind])
            del img
            if data_label[ind] == 1:
                imageLabels.append(0)
                imagePaths.append(data_image[ind])
                select_num1 += 1
            # if data_label[ind] == 77:
            #     imageLabels.append(0)
            if select_num1 > select_total:
                break
        except:
            continue

    return imagePaths, imageLabels


def get_error_list(error_file_path, select_num=3000):
    error_file = pd.read_csv(error_file_path, sep='\t')
    error_dict = Counter(error_file['primary_labels'])
    sorted_error_dict = dict(
        sorted(error_dict.items(), key=lambda d: d[1], reverse=True))
    keys_10 = list(sorted_error_dict.keys())[:10]
    print('key_type: ', type(keys_10[0]))
    data = pd.read_csv('/data1/xusong/3c_clothes_classify/train_1225_add.csv')
    img_name = data['img_name']
    category_id_level4 = data['category_id_level4']
    primary_category_id = data['primary_category_id']

    num = 0
    imageLabels = []
    imagePaths = []
    for ind in range(len(img_name)):
        try:
            img = Image.open(img_name[ind])
            del img
            if int(category_id_level4[ind]) in keys_10:
                if int(primary_category_id[ind]) == 1:
                    imageLabels.append(0)
                    imagePaths.append(img_name[ind])
                    num += 1
                if int(primary_category_id[ind]) == 77:
                    imageLabels.append(1)
                    imagePaths.append(img_name[ind])
                    num += 1
        except:
            continue
        if num > select_num:
            break

    return imagePaths, imageLabels


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


def data_generator(height=224,
                   width=224,
                   channels=3,
                   batch_size=24,
                   imagePaths=None,
                   imageLabels=None,
                   isTest=False):
    if isTest:
        # 添加一个others类，确保测试类别数目与训练类别数目一致
        imagePaths = imagePaths + [imagePaths[-1]]
        imageLabels = imageLabels + [2]

        test_dict = {'filename': list(imagePaths), 'class': list(imageLabels)}
        test_df = pd.DataFrame(test_dict, dtype=str)
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            samplewise_std_normalization=True,
            preprocessing_function=keras.applications.resnet50.preprocess_input
        )
        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            x_col="filename",
            y_col="class",
            target_size=(height, width),
            batch_size=batch_size,
            seed=7,
            shuffle=False,
            # label的编码格式：这里为one-hot编码
            class_mode='categorical')

        return test_generator, None

    # imagePaths, imageLabels = get_train_data()
    (trainX, testX, trainY, testY) = train_test_split(imagePaths,
                                                      imageLabels,
                                                      test_size=0.20,
                                                      random_state=42)

    train_dict = {'filename': list(trainX), 'class': list(trainY)}
    test_dict = {'filename': list(testX), 'class': list(testY)}
    train_df = pd.DataFrame(train_dict, dtype=str)
    test_df = pd.DataFrame(test_dict, dtype=str)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        samplewise_std_normalization=True,
        # 给resnet50预处理图像的函数
        # preprocessing_function=keras.applications.resnet50.preprocess_input,
        preprocessing_function=keras.applications.resnet50.preprocess_input,
        # 图片旋转的角度范围，用来数据增强
        rotation_range=40,
        # 水平平移
        width_shift_range=0.2,
        # 高度平移
        height_shift_range=0.2,
        # 剪切强度
        shear_range=0.2,
        # 缩放强度
        zoom_range=0.2,
        # 水平翻转
        horizontal_flip=True,
        # 对图片做处理时需要填充图片，用最近的像素点填充
        fill_mode="nearest")

    # 读取训练数据
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="class",
        target_size=(height, width),
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        class_mode='categorical')

    # 读取验证数据
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        samplewise_std_normalization=True,
        preprocessing_function=keras.applications.resnet50.preprocess_input)

    valid_generator = valid_datagen.flow_from_dataframe(
        test_df,
        x_col="filename",
        y_col="class",
        target_size=(height, width),
        batch_size=batch_size,
        seed=7,
        shuffle=False,
        # label的编码格式：这里为one-hot编码
        class_mode='categorical')

    return train_generator, valid_generator


if __name__ == '__main__':
    batch_size = 24
    # imagePaths, imageLabels = get_train_data()
    shopline_data = '/data1/xusong/shopline_data/data/shopline_20201225.csv'
    xuanpin_data = '/data1/xusong/3c_clothes_classify/train_1225_add.csv'
    imagePaths, imageLabels = get_balanced_train_data(shopline_data,
                                                      xuanpin_data)
    # error_file_path = '/data1/xusong/shopline_data/data/error.txt_2w'
    # imagePaths, imageLabels = get_error_list(error_file_path)
    resnet50_fine_tune = get_pretrained_resnet()
    train_generator, valid_generator = data_generator(imagePaths=imagePaths,
                                                      imageLabels=imageLabels)

    train_num = train_generator.samples
    valid_num = valid_generator.samples

    resnet50_fine_tune = get_pretrained_resnet()
    # resnet50_fine_tune.load_weights(
    #     "./resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5")
    epochs = 5
    log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Specify the callback object
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    checkpoint_path = 'logs/resnet50_standard_cp.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
    # 数据是generator出来的，所以不能直接用fit
    history = resnet50_fine_tune.fit_generator(
        train_generator,
        steps_per_epoch=train_num // batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_num // batch_size,
        callbacks=[cp_callback, tensorboard_callback])

    resnet50_fine_tune.save('train_model_3class_resnet50_standard.h5')

    print(history.history.keys())
    plot_learning_curves(history, 'accuracy', epochs, 0, 1, name='accuracy')
    plot_learning_curves(history, 'loss', epochs, 1.5, 2.5, name="loss")
