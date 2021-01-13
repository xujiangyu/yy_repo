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
from requests.adapters import HTTPAdapter
s = requests.Session()
s.mount('http://', HTTPAdapter(max_retries=30))
s.mount('https://', HTTPAdapter(max_retries=30))

#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=True)  # If not set, all available CPUs will be used.
import hashlib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

cur_file_path, cur_file = os.path.split(os.path.realpath(__file__))

is_empty_str = lambda content: None == content or '' == content.strip()

headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36"
}
ip_pool = [  #(外网ip, 内网ip)
    # '以下代理机器已回收'
    ('149.129.129.18', '10.70.23.131'),
    ('149.129.136.51', '10.70.23.132'),
    ('149.129.188.225', '10.70.23.134'),
    ('149.129.179.93', '10.70.23.133'),
    ('149.129.179.222', '10.70.23.137'),
    ('149.129.188.133', '10.70.23.136'),
    ('149.129.188.103', '10.70.23.140'),
    ('149.129.188.235', '10.70.23.139'),
    ('149.129.137.23', '10.70.23.138'),
]


class DownloadImg(object):
    def __init__(self, save_imgs_path=None, img_overwrite=False):
        self.save_imgs_path = save_imgs_path
        self.img_overwrite = img_overwrite

    ## 下载图片
    def download_image(self, img_info):
        if type(img_info) == pd.Series:
            img_info = img_info
        else:
            img_info = img_info.iloc[0]

        try:
            img_url = img_info['图片'].strip()[1:-1]
            img_name_prefix = img_info['商品id']

            img_name = img_name_prefix + self.stringtomd5(img_url)

            img_path = self.save_imgs_path + f'/{img_name}.jpg'

            img_url_head_parts = (img_url.split('?', maxsplit=1)[0]).rsplit(
                '.', maxsplit=1)
            img_form = img_url_head_parts[1].lower()

            if img_form in ['png', 'bmp', 'webp']:
                img_path = self.save_imgs_path + f'/{img_name}.{img_form}'
            if os.path.exists(img_path):
                print(
                    f'-->> {img_name} image url= {img_url} has been downloaded before!'
                )
                return img_path

            # r = requests.get(img_url, proxies=self.__get_ip_proxy(net_flag=1), headers=headers, timeout=(6.1, 30.1)) # 使用代理
            r = requests.get(img_url,
                             headers=headers,
                             timeout=(6.1, 30.1),
                             verify=False)
            img = r.content

            # 注意默认直接保存为jpg格式图片，因为有些链接可能存在跳转，原始的url不含图片格式
            with open(img_path, 'wb') as f:
                f.write(img)
                f.close()

            if img_form in ['png', 'bmp', 'webp']:
                img_path = self.__trans_jpg(img_path)

        except BaseException as e:
            print(
                f'-->> {img_name} image url= {img_url} download failed! Because: {e}'
            )

        if is_empty_str(img_path) or img_path.rsplit(
                '.', maxsplit=1)[1] not in ('jpg'):
            img_path = ''  #下载的图片格式不合要求，路径置为空
        print(f'-->> {img_name} image url= {img_url} has been downloaded!')
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
    def stringtomd5(self, str0):
        signaturemd5 = hashlib.md5()
        signaturemd5.update(str0.encode('utf8'))

        return signaturemd5.hexdigest()


def main(saved_imgs_path='shopline_2/'):
    # data_pro_path = "save_result.csv"
    data = pd.ExcelFile('./2.xlsx')
    choose_sheet = data.parse('交付数据2W')
    new_data = pd.concat([
        choose_sheet['商品id'], choose_sheet['图片'], choose_sheet['一级'],
        choose_sheet['二级'], choose_sheet['三级'], choose_sheet['四级'],
        choose_sheet['五级']
    ],
                         axis=1)
    # df = pd.read_csv("/data/seven/workspace/data/shopline/shopline_2020-12-08",
    #                  sep="\t",
    #                  header=None,
    #                  names=[
    #                      'merchant_id', 'product_id', 'product_image_urls',
    #                      'product_title_zh', 'product_title_en',
    #                      'product_title_zh_hant'
    #                  ])

    df_sel = new_data[np.logical_not(new_data['商品id'].isna())]

    dfc = df_sel.copy()

    dfc['商品id'] = df_sel['商品id'].apply(lambda x: f'{x}_')

    download_img = DownloadImg(save_imgs_path=saved_imgs_path,
                               img_overwrite=False)

    print(f'-->> Image download begin')
    begin = time.time()
    df_sel['img_name'] = dfc[['图片', '商品id'
                              ]].parallel_apply(download_img.download_image,
                                                axis=1)
    avg_time = (time.time() - begin) / df_sel.shape[0]
    print(f'<<-- Image download consumes {avg_time}s')
    dfc = df_sel.copy()
    df_sel = dfc[dfc['img_name'] != '']

    print('Work all done!!!')


if __name__ == '__main__':
    main()
