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
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.applications.resnet_v2 as resnetv2

import bert
from bert.loader import load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from image_augmentation import augument_image, is_valid_image
from PIL import Image
from imgaug import augmenters as iaa

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        print("WARNING!!! nothing to do!!!")
        return image


def load_image(img_names, img_dir, ima_size=224, is_train=True):
    x = []
    count = 0
    for img_name in img_names:
        img_name = os.path.join(img_dir, img_name)
        if not is_valid_image(img_name):
            x.append(np.zeros([ima_size, ima_size, 3]))
            count += 1
            print('!!')
            continue
        try:
            img = Image.open(img_name)
            if img.mode != 'RGB':
                img = img.convert("RGB")
            if is_train:
                img = img.resize((256, 256))
                img = random_crop(img, [224, 224], padding=10)
                img = np.asarray(img)
            else:
                # img = img.resize((ima_size, ima_size))
                img = img.resize((256, 256))
                img = random_crop(img, [224, 224], padding=10)
                img = np.asarray(img)
            img = resnetv2.preprocess_input(img)
        except:
            img = np.zeros([ima_size, ima_size, 3])
            print('!!')
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
    #cls_out = keras.layers.Dropout(0.3)(cls_out)
    x1 = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    # x1 = keras.layers.Dropout(0.3)(x1)

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
    # x2 = keras.layers.Dropout(0.3)(x2)

    if is_text:
        x = tf.concat([x1], 1)
    elif is_image:
        x = tf.concat([x2], 1)
    else:
        x = tf.concat([x1, x2], 1)
        # x = 0.5*x1 + x2
    #x = keras.layers.Dropout(0.3)(x)

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


def main():
    IS_TEXT = False
    IS_IMAGE = False
    IS_JUST_CLOTH = True
    IS_DENSENET = False
    test_data_file = 'test_data_1101.csv'
    out_test_file = 'out_data_1101_2.csv'

    batch_size = 1
    ima_size = 224
    max_seq_len = 64

    data_dir = r"/data1/seven/workspace/models/cloth/data"
    data_file = r"train_1225_add.csv"  ##对应训练的文件,保持类别一致

    cur_dir = r"/data1/seven/workspace/models/cloth/bert_densenet"
    img_dir = r"/data1/seven/workspace/models/cloth/data"

    model_path = os.path.join(cur_dir, 'models/model_save_bertresnet_mul2')
    model_name = 'cp-0011.ckpt'

    bert_dir = os.path.join(cur_dir, r"uncased_L-12_H-768_A-12")
    bert_ckp = r"bert_model.ckpt"

    label_encoder = LabelEncoder()
    df = pd.read_csv(os.path.join(data_dir, data_file))
    if IS_JUST_CLOTH:
        df = df[df['primary_category_id'] == 77]

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

    num_classes = [len(e.classes_) for e in label_encoder]
    tokenizer = FullTokenizer(os.path.join(bert_dir, "vocab.txt"), True)

    #
    df_test = pd.read_csv(os.path.join(data_dir, test_data_file))
    df_test = df_test.drop_duplicates(['biz_id', 'dt'])
    df_test['is_img_valid'] = df_test.apply(
        lambda x: 1
        if is_valid_image(os.path.join(img_dir, str(x['img_name']))) else 0,
        axis=1)
    df_test = df_test[df_test['is_img_valid'] == 1]
    if IS_DENSENET:
        np_image_x = load_image_1(list(df_test['img_name']), img_dir, ima_size,
                                  False)
    else:
        np_image_x = load_image(list(df_test['img_name']), img_dir, ima_size,
                                False)
    np_text_x = text_2_id(tokenizer, list(df_test['title']), max_seq_len)

    model1 = build_model(bert_dir, bert_ckp, 'imagenet', max_seq_len, ima_size,
                         num_classes, IS_TEXT, IS_IMAGE, IS_DENSENET)
    print("--model_path:", os.path.join(model_path, model_name))
    model1.load_weights(os.path.join(model_path, model_name))

    np_image_x = np.asarray(np_image_x).astype(np.float32)
    np_text_x = np.asarray(np_text_x).astype(np.float32)
    res = model1.predict([np_text_x, np_image_x])
    res = res[2]
    max_index = np.argmax(res, axis=-1)
    max_prob = np.max(res, axis=-1)

    df_test['probs'] = list(res)
    df_test['pred'] = label_encoder[2].inverse_transform(
        max_index.reshape(-1, ))
    df_test['prob'] = list(max_prob)
    df_test['is_true'] = df_test.apply(
        lambda x: 1 if str(x['pred']) == str(x['category_id_level4']) else 0,
        axis=1)

    print(np_image_x.shape)
    print(np_text_x.shape)
    print(len(df_test[df_test['is_true'] == 1].index), len(df_test.index),
          len(df_test[df_test['is_true'] == 1].index) / len(df_test.index))

    df_test.to_csv(out_test_file, index=False)

    with open('category.id', 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(label_encoder[2].classes_)))


if __name__ == '__main__':
    main()
