# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:02:10 2020

@author: Administrator
"""

import os

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
from image_augmentation import augument_image, is_valid_image
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    model.build(input_shape=[(None, max_seq_len), (None, ima_size, ima_size,
                                                   3)])

    # model.load_weights(densenet_checkpoint_path, by_name=True, skip_mismatch=True)
    load_stock_weights(bertLayer, os.path.join(bert_checkpoint_dir, bert_ckp))

    model.compile(optimizer=keras.optimizers.Adam(0.00005),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  loss_weights=[1., 1., 1.],
                  metrics=['accuracy'])
    model.summary()

    return model


def get_train_data(data_dir,
                   data_file,
                   tokenizer,
                   label_encoder,
                   max_seq_len,
                   img_dir,
                   ima_size,
                   is_just_cloth=True,
                   is_text=False,
                   is_image=False):
    df = pd.read_csv(os.path.join(data_dir, data_file))
    if is_image:
        df = df.drop_duplicates(['img_name'])
        df['is_img_valid'] = df.apply(
            lambda x: 1
            if is_valid_image(os.path.join(img_dir, x['img_name'])) else 0,
            axis=1)
        df = df[df['is_img_valid'] == 1]
    elif is_text:
        df = df.drop_duplicates(['title'])
    else:
        df['is_img_valid'] = df.apply(
            lambda x: 1
            if is_valid_image(os.path.join(img_dir, x['img_name'])) else 0,
            axis=1)
        df = df[df['is_img_valid'] == 1]

    if 'is_train' not in df.columns:
        test_id = random.sample(list(set(list(df['biz_id']))),
                                int(len(list(df['biz_id'])) * 0.2))
        df['is_train'] = df.apply(lambda x: 0 if x['biz_id'] in test_id else 1,
                                  axis=1)
        df = df.sample(frac=1)
        df.to_csv(os.path.join(data_dir, data_file))
    if is_just_cloth:
        df = df[df['primary_category_id'] == 77]
    '''
    d1 = df.copy(deep=True)
    d1['title'] = ""
    df = df.append(d1, ignore_index=True)
    df = df.sample(frac=1)
    '''
    df['category_id_level2'] = df.apply(lambda x: str(x['category_id_level2']),
                                        axis=1)
    df['category_id_level3'] = df.apply(lambda x: str(x['category_id_level3']),
                                        axis=1)
    df['category_id_level4'] = df.apply(lambda x: str(x['category_id_level4']),
                                        axis=1)
    df = df.fillna('')
    df_train = df[df['is_train'] == 1]
    df_test = df[df['is_train'] == 0]
    print('df_train, df_test,', df_train.shape, df_test.shape)

    num_classes = [0, 0, 0]
    num_classes[0] = len(list(label_encoder[0].classes_))
    num_classes[1] = len(list(label_encoder[1].classes_))
    num_classes[2] = len(list(label_encoder[2].classes_))

    train_y_2 = label_encoder[0].transform(list(
        df_train['category_id_level2']))
    test_y_2 = label_encoder[0].transform(list(df_test['category_id_level2']))
    train_y_3 = label_encoder[1].transform(list(
        df_train['category_id_level3']))
    test_y_3 = label_encoder[1].transform(list(df_test['category_id_level3']))
    train_y_4 = label_encoder[2].transform(list(
        df_train['category_id_level4']))
    test_y_4 = label_encoder[2].transform(list(df_test['category_id_level4']))

    train_x = text_2_id(tokenizer, list(df_train['title']), max_seq_len)
    test_x = text_2_id(tokenizer, list(df_test['title']), max_seq_len)
    train_x_image = np.array(list(df_train['img_name']))
    test_x_image = np.array(list(df_test['img_name']))

    return train_x_image, test_x_image, train_x, test_x, \
            [train_y_2, train_y_3, train_y_4], \
            [test_y_2, test_y_3, test_y_4], num_classes


class Data_Generator(keras.utils.Sequence):
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
            #if self.is_train:
            #    mask_text = random.sample(list(range(len(np_text_x))), int(len(np_text_x)*0))
            #    np_text_x[mask_text, :] = 0

            if self.is_densenet:
                np_image_x = load_image_1(batch_x_image, self.img_dir,
                                          self.ima_size, self.is_train)
            else:
                np_image_x = load_image(batch_x_image, self.img_dir,
                                        self.ima_size, self.is_train)

        return [np_text_x, np_image_x], [np_y_2, np_y_3, np_y_4]


def main():
    IS_INIT = False
    IS_JUST_CLOTH = True

    IS_TEXT = False
    IS_IMAGE = False
    IS_DENSENET = False

    # cur_dir = r"/data1/seven/workspace/models/cloth/bert_densenet"
    cur_dir = '/data1/xusong/clothes_classify/bert_densenet'
    img_dir = r'/data1/lisiguang/clothing_classifier_en/bert/data/img/amazon_independweb_imgs'

    model_path = os.path.join(cur_dir, 'models/model_save_bertresnet_mul0')

    num_epoc = 1
    batch_size = 64
    img_size = 224
    max_seq_len = 64

    # data_dir = r"/data1/seven/workspace/models/cloth/data"
    data_dir = '/data1/xusong/clothes_classify/data'
    data_file = r"train_1225.csv"

    bert_dir = os.path.join(cur_dir, r"uncased_L-12_H-768_A-12")
    bert_ckp = r"bert_model.ckpt"

    densenet_dir = os.path.join(cur_dir, r"model_save_densenet")
    densenet_ckp = r"final_weights.h5"

    checkpoint_path = os.path.join(model_path, 'cp-{epoch:04d}.ckpt')

    df = pd.read_csv(os.path.join(data_dir, data_file))
    df['category_id_level2'] = df.apply(lambda x: str(x['category_id_level2']),
                                        axis=1)
    df['category_id_level3'] = df.apply(lambda x: str(x['category_id_level3']),
                                        axis=1)
    df['category_id_level4'] = df.apply(lambda x: str(x['category_id_level4']),
                                        axis=1)

    label_encoder = [LabelEncoder(), LabelEncoder(), LabelEncoder()]
    label_encoder[0].fit(list(df['category_id_level2']))
    label_encoder[1].fit(list(df['category_id_level3']))
    label_encoder[2].fit(list(df['category_id_level4']))

    tokenizer = FullTokenizer(os.path.join(bert_dir, "vocab.txt"), True)
    x_train_img, x_test_img, x_train_text, x_test_text, y_train, y_test, num_classes = \
                    get_train_data(data_dir, data_file, tokenizer, label_encoder, max_seq_len,
                                   img_dir, img_size, IS_JUST_CLOTH, IS_TEXT, IS_IMAGE)
    train_batch_generator = Data_Generator(x_train_img,
                                           x_train_text,
                                           y_train,
                                           batch_size,
                                           img_size,
                                           img_dir,
                                           IS_TEXT,
                                           IS_IMAGE,
                                           IS_DENSENET,
                                           is_train=True)
    test_batch_generator = Data_Generator(x_test_img,
                                          x_test_text,
                                          y_test,
                                          batch_size,
                                          img_size,
                                          img_dir,
                                          IS_TEXT,
                                          IS_IMAGE,
                                          IS_DENSENET,
                                          is_train=False)

    ## 模型训练和保存
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        period=1,
    )

    model = build_model(bert_dir, bert_ckp,
                        os.path.join(densenet_dir, densenet_ckp), max_seq_len,
                        img_size, num_classes, IS_TEXT, IS_IMAGE, IS_DENSENET)

    if IS_INIT:
        latest = tf.train.latest_checkpoint(model_path)
        model.load_weights(latest)
    '''
    history = model.fit(train_batch_generator,
                        steps_per_epoch=int(y_train.shape[0] // batch_size)-1,
                        epochs=num_epoc, 
                        verbose=1,
                        validation_data=test_batch_generator,
                        callbacks=[cp_callback], )
    '''
    history = model.fit_generator(
        train_batch_generator,
        steps_per_epoch=int(x_train_img.shape[0] // batch_size) - 1,
        epochs=num_epoc,
        verbose=1,
        validation_data=test_batch_generator,
        callbacks=[cp_callback],
        workers=4,
    )

    # model.save_weights(os.path.join(model_path, "final_weights.h5"))
    with open(os.path.join(model_path, 'history.pkl'), 'wb') as f:
        pkl.dump(history.history, f)


if __name__ == '__main__':
    main()
