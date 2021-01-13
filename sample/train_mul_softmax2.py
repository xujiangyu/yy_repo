# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:02:10 2020

@author: Administrator
"""

import os
import sys

sys.path.append('../')

import random
import numpy as np
import pandas as pd
import pickle as pkl
import cv2

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.densenet as densenet
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Softmax
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.applications.resnet_v2 as resnetv2

import bert
from bert.loader import load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import matplotlib.pyplot as plt
from image_augmentation import augument_image, is_valid_image
from PIL import Image
# from cleanlab.classification import LearningWithNoisyLabels
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def random_crop(image, crop_shape, padding=None):
    oshape = image.size
    if padding:
        oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
        img_pad.paste(image, (padding, padding))

        nh = random.randint(0, oshape_pad[0] - crop_shape[0])
        nw = random.randint(0, oshape_pad[1] - crop_shape[1])
        image_crop = img_pad.crop(
            (nh, nw, nh + crop_shape[0], nw + crop_shape[1]))
        return image_crop
    else:
        return image


def load_image(img_names, img_dir, ima_size=224, is_train=True):
    x = []
    count = 0
    for img_name in img_names:
        img_name = os.path.join(img_dir, img_name)
        if not is_valid_image(img_name):
            x.append(np.zeros([ima_size, ima_size, 3]))
            count += 1
            continue
        try:
            img = Image.open(img_name).convert('RGB')
            if is_train:
                img = img.resize((256, 256))
                img = random_crop(img, [ima_size, ima_size], padding=10)
                img = np.asarray(img)
            else:
                # img = img.resize((ima_size, ima_size))
                img = img.resize((256, 256))
                img = random_crop(img, [224, 224], padding=10)
                img = np.asarray(img)
            img = resnetv2.preprocess_input(img)
        except:
            img = np.zeros([ima_size, ima_size, 3])
        x.append(img)
    return np.array(x)


def load_image_1(img_names, img_dir, ima_size=224, is_train=True):
    x = []
    count = 0
    for img_name in img_names:
        img_name = os.path.join(img_dir, img_name)
        if not is_valid_image(img_name):
            x.append(np.zeros([ima_size, ima_size, 3]))
            count += 1
            continue
        try:
            img = Image.open(img_name).convert('RGB')
            if 1:
                img = np.asarray(img)
                img = augument_image(img, is_show=False)
            else:
                img = img.resize((ima_size, ima_size))
                img = np.asarray(img)
            img = densenet.preprocess_input(img)
        except:
            img = np.zeros([ima_size, ima_size, 3])
        x.append(img)
    return np.array(x)


def text_2_id(tokenizer, texts, max_seq_len):
    x = []
    for text in texts:
        tokens = tokenizer.tokenize(str(text))
        tokens = ['[CLS]'] + tokens
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids += [0] * max_seq_len
        token_ids = token_ids[:max_seq_len]
        x.append(token_ids)
    return np.array(x)


def build_model(
    bert_checkpoint_dir,
    bert_ckp,
    densenet_checkpoint_path,
    max_seq_len,
    ima_size,
    num_classes,
    is_text=False,
    is_image=False,
    is_densenet=True,
):
    inputs_bert = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_params = bert.loader.params_from_pretrained_ckpt(bert_checkpoint_dir)
    bertLayer = bert.BertModelLayer.from_params(bert_params)
    hidden_out = bertLayer(inputs_bert)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(hidden_out)
    cls_out = keras.layers.Dropout(0.3)(cls_out)
    x1 = keras.layers.Dense(units=768, activation="tanh")(cls_out)

    inputs_densenet = Input(shape=(ima_size, ima_size, 3))
    if is_densenet:
        # densenet.DenseNet121()
        image_model = densenet.DenseNet201(
            include_top=False,
            weights='imagenet',
            input_shape=(ima_size, ima_size, 3),
        )
    else:
        image_model = resnetv2.ResNet101V2(
            include_top=False,
            weights='imagenet',
            input_shape=(ima_size, ima_size, 3),
        )

    x2 = image_model(inputs_densenet)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(768, activation='tanh')(x2)

    if is_text:
        x = tf.concat([x1], 1)
    elif is_image:
        x = tf.concat([x2], 1)
    else:
        x = tf.concat([x1, x2], 1)
        # x = 0.5*x1 + x2
    x = keras.layers.Dropout(0.3)(x)

    # 多级分类实现
    # 未实现多级分类loss的一个重要部分：下级与上级预测是否为父子层级的loss实现
    level2_hidden = Dense(units=768, activation="relu")(x)
    level2_output = Dense(units=num_classes[0],
                          activation="softmax")(level2_hidden)

    level3_hidden = Dense(units=768, activation="relu")(x)
    level3_output = Dense(units=num_classes[1], activation="softmax")(
        tf.concat([level2_hidden, level3_hidden], 1))

    level4_hidden = Dense(units=768, activation="relu")(x)
    level4_output = Dense(units=num_classes[2], activation="softmax")(
        tf.concat([level2_hidden, level3_hidden, level4_hidden], 1))

    model = Model(inputs=[inputs_bert, inputs_densenet],
                  outputs=[level2_output, level3_output, level4_output])
    # model = Model(inputs=[inputs_bert, inputs_densenet],
    #               outputs=[level2_output, level3_output])
    model.build(input_shape=[(None, max_seq_len), (None, ima_size, ima_size,
                                                   3)])

    # 上面已经指定了densenet 加载imagenet预训练的参数， 在实验中发现， 从imagenet加载的参数最后效果会好一点点。
    # 所以这里就注释掉了。
    # model.load_weights(densenet_checkpoint_path, by_name=True, skip_mismatch=True)

    load_stock_weights(bertLayer, os.path.join(bert_checkpoint_dir, bert_ckp))

    model.compile(
        optimizer=keras.optimizers.Adam(0.00005),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # 调整loss_weights对结果的影响
        loss_weights=[1, 1., 1],
        metrics=['accuracy'])
    model.summary()

    return model


def hierachial_loss(alpha=0.7, beta=0.3):
    """新增层级损失部分
    @param alpha: 叶子节点loss不对的损失
    @param beta: 层级之间不为父子关系的损失比重
    """


def get_train_data(data_dir,
                   data_file,
                   tokenizer,
                   label_encoder,
                   max_seq_len,
                   img_dir,
                   ima_size,
                   is_just_fushi=True,
                   is_text=False,
                   is_image=False):
    df = pd.read_csv(os.path.join(data_dir, data_file))

    if is_image and is_text:
        print("is_image and is_text can not be true at same time")
    if is_image:
        # 仅使用图片，对图片去重
        df = df.drop_duplicates(['img_name'])
        df['is_img_valid'] = df.apply(
            lambda x: 1
            if is_valid_image(os.path.join(img_dir, x['img_name'])) else 0,
            axis=1)
        df = df[df['is_img_valid'] == 1]
    elif is_text:
        # 仅使用文本，对文本去重
        df = df.drop_duplicates(['title'])
    else:
        # 混合文本与图片
        df['is_img_valid'] = df.apply(lambda x: 1 if is_valid_image(
            os.path.join(img_dir, str(x['img_name']))) else 0,
                                      axis=1)
        df = df[df['is_img_valid'] == 1]

    if is_just_fushi:
        df = df[df['primary_category_name'] == 'clothes']

    x_image, x_text, label, num_classes = label_and_feature_transform(
        df, label_encoder, tokenizer, max_seq_len, is_image, is_text, ima_size)
    return x_image, x_text, label, num_classes


def label_and_feature_transform(df,
                                label_encoder,
                                tokenizer,
                                max_seq_len,
                                is_image,
                                is_text,
                                ima_size,
                                level=3):
    num_classes = [0] * level
    label = [0] * level
    for i in range(level):
        df[f'category_id_level{i + 2}'] = df.apply(
            lambda x: str(x[f'category_id_level{i + 2}']), axis=1)
        num_classes[i] = len(list(label_encoder[i].classes_))
        label[i] = label_encoder[i].transform(
            list(df[f'category_id_level{i + 2}']))

    df = df.fillna('')
    if is_text:
        # 仅文本
        x_text = text_2_id(tokenizer, list(df['title']), max_seq_len)
        x_image = np.array([0] * ima_size)
    elif is_image:
        # 仅图片
        x_text = np.array([[0] * max_seq_len] * len(df['title']))
        x_image = np.array(list(df['img_name']))
    else:
        x_text = text_2_id(tokenizer, list(df['title']), max_seq_len)
        x_image = np.array(list(df['img_name']))

    return x_image, x_text, label, num_classes


class Data_Generator(keras.utils.Sequence):
    """Sequence是对Dataset等流式数据集的高级封装，可以保证多卡训练时，一次epoch中，每个example只被输入网络一次"""
    def __init__(self,
                 np_image,
                 np_text,
                 labels,
                 batch_size,
                 ima_size,
                 img_dir,
                 is_text=False,
                 is_image=False,
                 is_densenet=False,
                 is_train=True):
        self.np_image = np_image
        self.np_text = np_text
        self.labels = labels
        self.batch_size = batch_size
        self.ima_size = ima_size
        self.img_dir = img_dir
        self.is_text, self.is_image = is_text, is_image
        self.is_densenet = is_densenet
        self.is_train = is_train

    def __len__(self):
        return (np.ceil(len(self.np_text) / float(self.batch_size))).astype(
            np.int)

    def __getitem__(self, idx):
        # 多级类目切分batch
        np_y_2 = np.array(self.labels[0][idx * self.batch_size:(idx + 1) *
                                         self.batch_size])
        np_y_3 = np.array(self.labels[1][idx * self.batch_size:(idx + 1) *
                                         self.batch_size])
        np_y_4 = np.array(self.labels[2][idx * self.batch_size:(idx + 1) *
                                         self.batch_size])

        if self.is_text:
            np_text_x = np.array(self.np_text[idx * self.batch_size:(idx + 1) *
                                              self.batch_size])
            np_image_x = np.zeros(
                [self.batch_size, self.ima_size, self.ima_size, 3])

        elif self.is_image:
            batch_x_image = self.np_image[idx * self.batch_size:(idx + 1) *
                                          self.batch_size]
            np_text_x = np.array(self.np_text[idx * self.batch_size:(idx + 1) *
                                              self.batch_size])

            if self.is_densenet:
                np_image_x = load_image_1(batch_x_image, self.img_dir,
                                          self.ima_size, self.is_train)
            else:
                np_image_x = load_image(batch_x_image, self.img_dir,
                                        self.ima_size, self.is_train)
        else:
            batch_x_image = self.np_image[idx * self.batch_size:(idx + 1) *
                                          self.batch_size]
            np_text_x = np.array(self.np_text[idx * self.batch_size:(idx + 1) *
                                              self.batch_size])
            # if self.is_train:
            #     # 可通过调整文本采样后mask为0的比例，来提升图片部分的话语权
            #     mask_text = random.sample(list(range(len(np_text_x))), int(len(np_text_x)*0))
            #     np_text_x[mask_text, :] = 0

            if self.is_densenet:
                np_image_x = load_image_1(batch_x_image, self.img_dir,
                                          self.ima_size, self.is_train)
            else:
                np_image_x = load_image(batch_x_image, self.img_dir,
                                        self.ima_size, self.is_train)

        return [np_text_x, np_image_x], [np_y_2, np_y_3, np_y_4]


def label_encoder_fit(data_dir, data_name_list, level=3):
    """合并3个数据集，获得各级label的编码
       @param level: 除顶层之外的层级。3c只有2层，而服饰有3层
    """

    # 将多级类目label由整形转为字符串，然后使用Sklearn进行label归一化编码
    dfs = [
        pd.read_csv(os.path.join(data_dir, data_file))
        for data_file in data_name_list
    ]
    df = pd.concat(dfs)
    label_encoder = [LabelEncoder() for i in range(level)]
    for i in range(2, level + 2):
        df[f'category_id_level{i}'] = df.apply(
            lambda x: str(x[f'category_id_level{i}']), axis=1)
        label_encoder[i - 2].fit(list(df[f'category_id_level{i}']))
    return label_encoder


def plot(history, index):
    """画出validate loss 和 train loss"""
    plt.plot(history.history[index])
    plt.plot(history.history['val_' + index])
    plt.xlabel('Epochs')
    plt.ylabel(index)
    plt.legend([index, 'val_' + index])
    plt.show()


def main():
    # 整体模型的参数，是否从之前训练的某一版加载
    IS_INIT = False
    # 是否仅适用服饰的数据进行训练（如果为False，则加入3C）
    IS_JUST_FUSHI = False

    # 是否仅适用文字特征
    IS_TEXT = False
    # 是否仅适用图片特征，注意不能与is_text同时为True
    IS_IMAGE = False
    # 图片特征部分是否使用Densenet，False的话则采用resnet
    IS_DENSENET = False

    # 指定项目的root目录
    # cur_dir = r"/data1/seven/workspace/models/cloth/bert_densenet"
    cur_dir = '/data1/xusong/clothes_classify/bert_densenet'
    model_path = os.path.join(cur_dir, 'models/img+txt/1105_test_1101in')

    num_epoc = 20
    batch_size = 128
    ima_size = 224
    max_seq_len = 32

    data_dir = r"/data1/lisiguang/clothing_classifier_en/bert/data"
    # data_file = r"data_title_dropduplicate.csv"
    # 默认使用多级分类的训练数据
    # train_data_file = "imgs_info_clothes_multi_level_1.csv"
    # train_data_file = "imgs_info_clothes_multi_level_2.csv"
    # train_data_file = "imgs_info_clothes_multi_level_1105_1.csv"
    train_data_file = "imgs_info_clothes_multi_level_1105_1101in.csv"
    # dev_data_file = 'val_data_result_1105_img.csv'
    # test_data_file = 'test.csv'
    # test_data_file = 'img_test.csv'
    # test_data_file = 'img_test_20201101.csv'
    test_data_file = 'test_notin_20201101_img_mul.csv'

    img_dir = r'/data1/lisiguang/clothing_classifier_en/bert/data/img/amazon_independweb_imgs'

    bert_dir = os.path.join(cur_dir, r"uncased_L-12_H-768_A-12")
    bert_ckp = r"bert_model.ckpt"

    # 直接从imgnet加载，效果会好一些
    # densenet_dir = os.path.join(cur_dir,"models/img/1130_test_img")
    # densenet_ckp = r"final_weights.h5"
    densenet_dir = ""
    densenet_ckp = ""

    checkpoint_path = os.path.join(model_path, 'cp-{epoch:04d}.ckpt')

    print("begin label encode")
    # label_encoder = label_encoder_fit(data_dir, [train_data_file, dev_data_file, test_data_file])
    # label_encoder = label_encoder_fit(data_dir, [train_data_file, test_data_file])
    # label_encoder = label_encoder_fit(data_dir, [train_data_file, dev_data_file])
    label_encoder = label_encoder_fit(data_dir, [train_data_file])
    print("label encode finished")

    print("begin process data")
    tokenizer = FullTokenizer(os.path.join(bert_dir, "vocab.txt"), True)
    x_train_ima, x_train_text, y_train, num_classes = \
        get_train_data(data_dir, train_data_file, tokenizer, label_encoder, max_seq_len,
                       img_dir, ima_size, IS_JUST_FUSHI, IS_TEXT, IS_IMAGE)
    # x_dev_ima, x_dev_text, y_dev, num_classes = \
    #     get_train_data(data_dir, dev_data_file, tokenizer, label_encoder, max_seq_len,
    #                    img_dir, ima_size, IS_JUST_FUSHI, IS_TEXT, IS_IMAGE)
    x_test_ima, x_test_text, y_test, num_classes = \
        get_train_data(data_dir, test_data_file, tokenizer, label_encoder, max_seq_len,
                       img_dir, ima_size, IS_JUST_FUSHI, IS_TEXT, IS_IMAGE)
    print("process data finished")

    # todo 可以进一步加入prefetch等Dataset高级特性来加快训练速度
    train_batch_generator = Data_Generator(x_train_ima,
                                           x_train_text,
                                           y_train,
                                           batch_size,
                                           ima_size,
                                           img_dir,
                                           IS_TEXT,
                                           IS_IMAGE,
                                           IS_DENSENET,
                                           is_train=True)
    # dev_batch_generator = Data_Generator(x_dev_ima, x_dev_text, y_dev, batch_size,
    #                                      ima_size, img_dir, IS_TEXT, IS_IMAGE, IS_DENSENET, is_train=False)
    test_batch_generator = Data_Generator(x_test_ima,
                                          x_test_text,
                                          y_test,
                                          batch_size,
                                          ima_size,
                                          img_dir,
                                          IS_TEXT,
                                          IS_IMAGE,
                                          IS_DENSENET,
                                          is_train=False)

    ## 模型训练和保存
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     period=1)

    model = build_model(bert_dir, bert_ckp,
                        os.path.join(densenet_dir, densenet_ckp), max_seq_len,
                        ima_size, num_classes, IS_TEXT, IS_IMAGE, IS_DENSENET)
    # rp = LearningWithNoisyLabels(clf=clf, seed=seed)
    if IS_INIT:
        latest = tf.train.latest_checkpoint(model_path)
        model.load_weights(latest)
    print(len(y_train[0]))
    print(len(y_test[0]))
    history = model.fit(
        train_batch_generator,
        # steps_per_epoch=int(y_train.shape[0] // batch_size)-1,
        epochs=num_epoc,
        verbose=1,
        validation_data=test_batch_generator,
        callbacks=[cp_callback],
    )

    # history = model.fit(train_batch_generator,
    #                     steps_per_epoch=len(y_train[0]) // batch_size - 1,
    #                     epochs=num_epoc,
    #                     verbose=1,
    #                     validation_data=dev_batch_generator,
    #                     validation_steps=len(y_dev[0]) // batch_size - 1,
    #                     callbacks=[cp_callback],
    #                     workers=4)

    # model.save_weights(os.path.join(model_path, "final_weights.h5"))
    with open(os.path.join(model_path, 'history.pkl'), 'wb') as f:
        pkl.dump(history.history, f)
    # plot(history, 'loss')
    # plot(history, 'accuracy')

    # print(f'test_set_size: {len(y_test[0])}')
    # model.evaluate(test_batch_generator,
    #               workers=1)


if __name__ == '__main__':
    main()
