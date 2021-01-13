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
import random
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


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


def check_image(img_name):
    try:
        img = Image.open(img_name)
        del img
        return True
    except:
        return False


def get_balanced_train_data(shopline_data,
                            xuanpin_data,
                            total=100000,
                            isTest=False):
    def extend_large_size(short_list, larg_size, test_size=20000):
        random.seed(2021)
        random.shuffle(short_list)
        # large_list = None
        a = larg_size
        b = len(short_list)
        if isTest:
            if b > a:
                return short_list[-test_size:]
            else:
                return []
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


def get_test_data(input_data_csv, select_num=1000):
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
                # 如果是衣服类别，标签设置为1
                testImageLabels.append(1)
                testImagePaths.append(csv_path[ind])
                start_num1 += 1
            elif int(csv_label[ind]) == 1:
                if ind > 20000:
                    # 如果是3C类别，标签设置为0
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


def data_generator(height=224,
                   width=224,
                   channels=3,
                   batch_size=24,
                   imagePaths=None,
                   imageLabels=None,
                   isTest=False,
                   isStandard=False):

    if isTest:
        # 添加一个others类，确保测试类别数目与训练类别数目一致
        if not imageLabels:
            newImagePaths = []
            for tmp_path in imagePaths:
                # print(tmp_path)
                # Image.open(tmp_path)
                if check_image(tmp_path):
                    newImagePaths.append(tmp_path)

            imagePaths = newImagePaths
            imageLabels = [0] * len(imagePaths)
            print('======', len(imagePaths), len(imageLabels))
        else:
            imagePaths = imagePaths + [imagePaths[-1]]
            imageLabels = imageLabels + [2]

        # if isStandard:
        #     imagePaths = [pixel_standard(li) for li in imagePaths]

        test_dict = {'filename': imagePaths, 'class': list(imageLabels)}

        # if isStandard:
        #     test_df = pd.DataFrame(test_dict)
        # else:
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
    # if isStandard:
    #     trainX = [pixel_standard(li) for li in trainX]
    #     testX = [pixel_standard(li) for li in testX]

    train_dict = {'filename': trainX, 'class': list(trainY)}
    test_dict = {'filename': testX, 'class': list(testY)}
    # if isStandard:
    #     train_df = pd.DataFrame(train_dict)
    #     test_df = pd.DataFrame(test_dict)
    # else:
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
    batch_size = 24
    # imagePaths, imageLabels = get_train_data()
    random.seed(2021)
    origin_data = pd.read_csv(
        '/data1/xusong/shopline_data/data/shopline_20201225.csv', dtype=str)
    filter_data = origin_data.loc[origin_data['一级'] == '电子产品']
    imagePaths = random.choices(list(filter_data['img_name']), k=1500)
    print(f'imagePaths len{len(imagePaths)}')
    # xuanpin_data = '/data1/xusong/3c_clothes_classify/train_1225_add.csv'
    # imagePaths, imageLabels = get_balanced_train_data(shopline_data,
    #                                                   xuanpin_data,
    #                                                   isTest=True)

    # input_csv = '/data1/seven/workspace/models/cloth/data/train_1225.csv'
    test_datagen, _ = data_generator(imagePaths=imagePaths, isTest=True)
    # test_datagen.filenames
    # test_datagen.next()[0][1]
    # print(test_datagen.)

    model_weight_path = '/data1/xusong/shopline_data/train_model_3class_resnet_balance_standard.h5'
    resnet50_fine_tune = get_pretrained_resnet()
    resnet50_fine_tune.load_weights(model_weight_path)

    print("original test labels: ", len(test_datagen.classes))

    predIdxs = resnet50_fine_tune.predict(test_datagen, batch_size=batch_size)

    predict_labels = np.argmax(predIdxs, axis=1)

    # predict_max = np.max(predIdxs, axis=1)
    # thres = 0.8

    # flag = (predict_max < thres) * (predict_labels == 0)
    # predict_labels[flag] = 2

    # product_dict = {0: '3C', 1: 'clothes', 2: 'others'}
    # pre_csv = pd.DataFrame({
    #     'img_name': test_datagen.filenames,
    #     'predict_num': predict_labels
    # })
    # pre_csv['predict_label'] = pre_csv.apply(
    #     lambda x: product_dict[x['predict_num']], axis=1)

    # pre_csv.to_csv(
    #     '/data1/xusong/shopline_data/data/shopline_20210108_predict/20210108_predict.csv',
    #     index=False)

    # for ind in range(len(predict_labels)):
    #     if predict_labels[ind] == 0:
    #         cmd = 'cp ' + test_datagen.filenames[
    #             ind] + ' ' + '/data1/xusong/shopline_data/data/shopline_20210108_predict/3C_0/'
    #     if predict_labels[ind] == 1:
    #         cmd = 'cp ' + test_datagen.filenames[
    #             ind] + ' ' + '/data1/xusong/shopline_data/data/shopline_20210108_predict/clothes_1/'
    #     if predict_labels[ind] == 2:
    #         cmd = 'cp ' + test_datagen.filenames[
    #             ind] + ' ' + '/data1/xusong/shopline_data/data/shopline_20210108_predict/others_2/'
    #     os.popen(cmd)

    print(
        tf.math.confusion_matrix(test_datagen.classes,
                                 predict_labels,
                                 num_classes=3))
    # print(classification_report(test_datagen.classes, predict_labels))
    print("error_id: ", test_datagen.classes != predict_labels)

    # fout = open('error.txt_2w_add', 'w')
    # fout.writelines("primary_labels\t" + "predicted_labels\t" + "img_name\n")
    # for ind in range(len(test_datagen.classes)):
    #     if test_datagen.classes[ind] != predict_labels[ind]:
    #         fout.writelines(
    #             str(test_datagen.classes[ind]) + '\t' +
    #             str(predict_labels[ind]) + '\t' + test_datagen.filenames[ind] +
    #             '\n')

    # fout = open('error.txt', 'w')
    # res = np.array(
    #     test_datagen.filenames)[test_datagen.classes != predict_labels]

    # for item in res:
    #     fout.writelines(item + '\n')
    # np.savetxt("error.txt", res)
    # fout.close()

    # fin = open('error.txt', 'r')
    # lines = fin.readlines()
    # for li in lines:
    #     cmd = 'cp ' + li.strip() + ' ' + './error_images/'
    #     os.popen(cmd)
