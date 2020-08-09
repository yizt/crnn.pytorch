# -*- coding: utf-8 -*-
"""
 @File    : rest_test.py
 @Time    : 2020/4/6 上午10:12
 @Author  : yizuotian
 @Description    :
"""

import base64

import requests

img_path = './images/horizontal-002.jpg'

with open(img_path, 'rb') as fp:
    img_bytes = fp.read()

img = base64.b64encode(img_bytes).decode()
data = {'img': img}

r = requests.post("http://localhost:5000/crnn", json=data)

print(r.json()['text'])
