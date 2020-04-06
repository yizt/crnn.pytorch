# -*- coding: utf-8 -*-
"""
 @File    : rest.py
 @Time    : 2020/4/6 上午9:39
 @Author  : yizuotian
 @Description    : restful服务
"""

import base64

import cv2
import numpy as np
import tornado.httpserver
import tornado.wsgi
from flask import Flask, request

app = Flask(__name__)


def pre_process_image(image, h, w):
    """

    :param image: [H,W]
    :param h: 图像高度
    :param w: 图像宽度
    :return:
    """
    if h != 32 and h < w:
        new_w = int(w * 32 / h)
        image = cv2.resize(image, (new_w, 32))
    if w != 32 and w < h:
        new_h = int(h * 32 / w)
        image = cv2.resize(image, (32, new_h))

    if h < w:
        image = np.array(image).T  # [W,H]
    image = image.astype(np.float32) / 255.
    image -= 0.5
    image /= 0.5
    image = image[np.newaxis, np.newaxis, :, :]  # [B,C,W,H]
    return image


@app.route('/crnn', methods=['POST'])
def ocr_rest():
    """
    :return:
    """

    img = base64.decodebytes(request.form.get('img').encode())
    img = np.frombuffer(img, dtype=np.uint8)
    shape = request.form.getlist('shape', type=int)
    img = img.reshape(shape)
    img = pre_process_image(img)
    cv2.imwrite('abc.jpg', img)

    return {'a': 'yes'}


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    start_tornado(app, 5000)
