import pandas as pd
import numpy as np
import requests
import os
import time
import random
from PIL import Image
from requests.adapters import HTTPAdapter
s = requests.Session()
from collections import Counter
import pickle
from PIL import Image
import hashlib
import sys
import urllib3

s.mount('http://', HTTPAdapter(max_retries=30))
s.mount('https://', HTTPAdapter(max_retries=30))
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


## 字符串编码
def stringtomd5(str0):
    signaturemd5 = hashlib.md5()
    signaturemd5.update(str0.encode('utf8'))
    return signaturemd5.hexdigest()


def captureImgPath(img_id, img_url):
    # global image_paths_dict
    if isinstance(img_url, np.float):
        return ' '
    code_id = img_id + '_' + stringtomd5(img_url)

    # if code_id in image_paths_dict.keys():
    #     return image_paths_dict[code_id]
    return code_id


def convert1():
    data11 = pd.ExcelFile(r'./1.xlsx')
    data22 = pd.ExcelFile(r'./2.xlsx')
    data33 = pd.ExcelFile(r'./3.xlsx')

    sheet11 = data11.parse('交付数据2W')
    sheet22 = data22.parse('交付数据2-4万（1105）')
    sheet33 = data33.parse('来源资料--5-9.5万')

    new_data11 = pd.concat([
        sheet11['商户id'], sheet11['商品id'], sheet11['标题'], sheet11['图片'],
        sheet11['一级'], sheet11['二级'], sheet11['三级'], sheet11['四级'],
        sheet11['五级'], sheet11['链接']
    ],
                           axis=1)

    new_data22 = pd.concat([
        sheet11['商户id'], sheet22['商品id'], sheet22['标题'], sheet22['图片'],
        sheet22['一级'], sheet22['二级'], sheet22['三级'], sheet22['四级'],
        sheet22['五级'], sheet22['链接']
    ],
                           axis=1)

    new_data33 = pd.concat([
        sheet11['商户id'], sheet33['商品id'], sheet33['标题'], sheet33['图片'],
        sheet33['一级'], sheet33['二级'], sheet33['三级'], sheet33['四级'],
        sheet33['五级'], sheet33['链接']
    ],
                           axis=1)

    new_data = pd.concat([new_data11, new_data22, new_data33], axis=0)

    image_folders = ['shopline_1', 'shopline_2', 'shopline_3', 'shopline_3_1']
    pwd = '/data1/xusong/shopline_data/'

    global image_paths_dict
    image_paths_dict = {}

    for folders in image_folders:
        list_files = os.listdir(pwd + folders)
        for files in list_files:
            key_str = files.split('.')[0]
            image_paths_dict[key_str] = pwd + folders + '/' + files

    def captureImgPath1(img_id, img_url):
        global image_paths_dict
        if isinstance(img_url, np.float):
            return ' '
        code_id = img_id + '_' + stringtomd5(img_url)
        if code_id in image_paths_dict.keys():
            return image_paths_dict[code_id]
        return ' '

    new_data['img_name'] = new_data.apply(
        lambda x: captureImgPath1(x['商品id'], x['图片']), axis=1)

    new_data.to_csv('data/shopline_20201225.csv', index=False)
    print(new_data.head())


def convert2():
    raw_csv = pd.read_csv('/tmp/shopline2.csv', dtype=str)
    raw_csv['img_name'] = raw_csv.apply(
        lambda x: '/data1/xusong/shopline_data/shopline_2021_01_08/' +
        captureImgPath(x['product_id'], x['img_url']) + '.jpg',
        axis=1)
    raw_csv.to_csv('shopline_2021_01_08_1.csv', index=False)


if __name__ == '__main__':
    # convert2()
    convert1()