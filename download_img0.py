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
pandarallel.initialize(nb_workers=10)
import hashlib
import sys

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# cur_file_path, cur_file = os.path.split(os.path.realpath(__file__))

is_empty_str = lambda content: None == content or '' == content.strip()

# input_file_path = "test_data_1225_ori.csv"
# #saved_imgs_path = "/data1/lisiguang/clothing_classifier_en/bert/data/img/amazon_independweb_imgs"
# saved_imgs_path = "/data1/seven/workspace/models/cloth/data/test_imgs"
# out_file_path = 'test_data_1225.csv'
# data_fail_path = 'imgs_download_fail.csv'

headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36"
}
ip_pool = [  #(外网ip, 内网ip)
    ('8.129.225.9', '10.31.4.10'),
]


def check_image(img_name):
    try:
        img = Image.open(img_name)
        del img
        return True
    except Exception as e:
        print("error info: ", e)
        return False


class DownloadImg(object):
    def __init__(self, save_imgs_path='.', img_overwrite=False):
        self.save_imgs_path = save_imgs_path
        self.img_overwrite = img_overwrite

    ## 下载图片
    def download_image(self, img_info):
        # global img_names_set
        try:
            img_url = img_info['图片']
        except:
            return ' '
        if not img_url or isinstance(img_url, np.float):
            print('error: ', img_url)
            return ' '

        img_name_prefix = img_info['商品id']
        img_name = img_name_prefix + '_' + self.__stringtomd5(img_url)

        img_path = self.save_imgs_path + f'/{img_name}.jpg'
        # if img_path in img_names_set:
        #     return ' '
        # 图片已经存在，而且不重写，则跳出
        # cwd = os.getcwd()
        flag = check_image(img_path)
        if not self.img_overwrite and os.path.exists(img_path) and flag:
            print("old!")

            # return cwd + '/' + img_path
            return img_path
        try:
            img_url_head_parts = (img_url.split('?', maxsplit=1)[0]).rsplit(
                '.', maxsplit=1)
            img_form = img_url_head_parts[1].lower()

            #r = requests.get(img_url, proxies=self.__get_ip_proxy(net_flag=1), headers=headers)
            r = requests.get(img_url, headers=headers,
                             timeout=(6.1, 30))  #connect和read的timeout
            img = r.content

            # 注意默认直接保存为jpg格式图片，因为有些链接可能存在跳转，原始的url不含图片格式
            if img_form in ['png', 'bmp', 'webp']:
                img_path = self.save_imgs_path + f'/{img_name}.{img_form}'
                with open(img_path, 'wb') as f:
                    f.write(img)
                    f.close()
                # 格式转化为jpg
                img_path = self.__trans_jpg(img_path)
            else:
                with open(img_path, 'wb') as f:
                    f.write(img)
                    f.close()
        except BaseException as e:
            print(
                f'-->> {img_name} image url= {img_url} download failed! Because: {e}'
            )
            img_path = ''

        if is_empty_str(img_path) or img_path.rsplit(
                '.', maxsplit=1)[1] not in ('jpg'):
            img_path = ''  #下载的图片格式不合要求，路径置为空

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


def process(saved_imgs_path, input_df):

    # df = pd.read_csv(input_file_path)
    download_img = DownloadImg(save_imgs_path=saved_imgs_path,
                               img_overwrite=False)
    input_df['new_img_name'] = input_df[['图片', '商品id']].parallel_apply(
        download_img.download_image, axis=1)

    #input_df[input_df['img_name'] != ''].to_csv('succes_1.csv', index=False)

    #input_df[input_df['img_name'] == ''].to_csv('fail_1.csv', index=False)


if __name__ == '__main__':
    data = pd.ExcelFile('./3.xlsx')

    # ready_file_path = '/data1/seven/workspace/data/shopline/results/images/'
    # img_name = set()
    # all_files = []
    # for _, _, tmp_files in os.walk(ready_file_path):
    #     all_files.extend(tmp_files)

    # img_names_set = set(all_files)
    # print(len(img_names_set), img_names_set.pop())

    # data = pd.read_csv('/tmp/shopline2.csv', sep=',', dtype=str)
    # print("csv_info: ", data.shape, data.columns)
    choose_sheet = data.parse('来源资料--5-9.5万')
    new_data = pd.concat([
        choose_sheet['商品id'], choose_sheet['图片'], choose_sheet['标题'],
        choose_sheet['一级'], choose_sheet['二级'], choose_sheet['三级'],
        choose_sheet['四级'], choose_sheet['五级']
    ],
                         axis=1)
    saved_imgs_path = 'shopline_3_1/'

    # # if len(sys.argv) == 4:
    # #     input_file_path = sys.argv[1]
    # #     saved_imgs_path = sys.argv[2]
    # #     out_file_path = sys.argv[3]
    process(saved_imgs_path, new_data)
    # data.to_csv('new_shopline.csv', index=False)
