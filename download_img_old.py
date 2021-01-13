# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 9:37
# @Author  : 葛莉
# @FileName: download_img1.py
# CodeBrief: 根据Amazon图片url，从网站下载保存图片，建立以3C和服饰为主的图库，需要在linux系统执行

import pandas as pd
import numpy as np
import requests
import os
import time
import random
from PIL import Image
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=24)
import hashlib
import sys

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

cur_file_path, cur_file = os.path.split(os.path.realpath(__file__))

is_empty_str = lambda content: None == content or '' == content.strip()

input_file_path = "train_1225_ori.csv"
saved_imgs_path = "/data1/lisiguang/clothing_classifier_en/bert/data/img/amazon_independweb_imgs"
out_file_path = 'train.csv'

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36"}
ip_pool = [#(外网ip, 内网ip)
    ('8.129.225.9', '10.31.4.10'),
]

class DownloadImg(object):
    def __init__(self, save_imgs_path=os.path.join(cur_file_path,'imgs'), img_overwrite=False):
        self.save_imgs_path = save_imgs_path
        self.img_overwrite = img_overwrite


    ## 下载图片
    def download_image(self, img_info):

        img_url = img_info['img_url'].strip()
        img_name_prefix = img_info['biz_id']
        img_name = img_name_prefix + '_' + self.__stringtomd5(img_url)

        img_path = self.save_imgs_path + f'/{img_name}.jpg'
        # 图片已经存在，而且不重写，则跳出
        if not self.img_overwrite and os.path.exists(img_path):
            print("old!")
            return img_path
        else:
            img_path = ''

        return img_path


    ## 随机获取代理ip，降低被封的风险
    def __get_ip_proxy(self, net_flag=0):
        ip = ip_pool[random.randrange(0, len(ip_pool))][net_flag]
        proxy_ip = f'http://{ip}:8128'
        proxy_ip1 = f'https://{ip}:8128'
        proxies = {'http': proxy_ip, 'https': proxy_ip1}
        return proxies

    ## 把其他格式（png）图片转化为jpg
    def __trans_jpg(self, img_path):
        try:
            str = img_path.rsplit('.', maxsplit=1)
            output_img_path = str[0] + '.jpg'
            im = Image.open(img_path).convert('RGB')
            im.save(output_img_path, 'jpeg')
            os.remove(img_path)
            return output_img_path
        except:
            return ''

    ## 字符串编码
    def __stringtomd5(self, str0):
        signaturemd5 = hashlib.md5()
        signaturemd5.update(str0.encode('utf8'))

        return signaturemd5.hexdigest()


def process():

    df = pd.read_csv(input_file_path)
    download_img = DownloadImg(save_imgs_path=saved_imgs_path, 
                               img_overwrite=False)
    df['img_name'] = df[['img_url', 'biz_id']].parallel_apply(download_img.download_image, axis=1)

    df[df['img_name'] != ''].to_csv(out_file_path, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        input_file_path = sys.argv[1]
        saved_imgs_path = sys.argv[2]
        out_file_path = sys.argv[3]
    process()
