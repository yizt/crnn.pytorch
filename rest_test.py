# -*- coding: utf-8 -*-
"""
 @File    : rest_test.py
 @Time    : 2020/4/6 上午10:12
 @Author  : yizuotian
 @Description    :
"""

import codecs

import cv2
import requests

img = cv2.imread('./images/horizontal-002.jpg', 0)
h, w = img.shape
data = {'img': codecs.encode(img.tostring(), 'base64'),  # 转为字节,并编码
        'shape': [h, w]}

r = requests.post("http://localhost:5000/crnn", data=data)

print(r.json()['text'])
