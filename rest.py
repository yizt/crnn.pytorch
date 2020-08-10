# -*- coding: utf-8 -*-
"""
 @File    : rest.py
 @Time    : 2020/4/6 上午9:39
 @Author  : yizuotian
 @Description    : restful服务
"""

import argparse
import base64
import itertools
import sys

import cv2
import numpy as np
import torch
import tornado.httpserver
import tornado.wsgi
from flask import Flask, request

import crnn
from config import cfg

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


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


def inference(image, h, w):
    """
    预测图像
    :param image: [H,W]
    :param h: 图像高度
    :param w: 图像宽度
    :return: text
    """
    image = torch.FloatTensor(image)
    image = image.to(device)

    if h > w:
        predict = v_net(image)[0].detach().cpu().numpy()  # [W,num_classes]
    else:
        predict = h_net(image)[0].detach().cpu().numpy()  # [W,num_classes]

    label = np.argmax(predict[:], axis=1)
    label = [alpha[class_id] for class_id in label]
    label = [k for k, g in itertools.groupby(list(label))]
    # label = ''.join(label).replace(' ', '')
    return label


@app.route('/crnn', methods=['POST'])
def ocr_rest():
    """
    :return:
    """
    img_bytes = base64.b64decode(request.json['img'].encode())
    img = cv2.imdecode(np.frombuffer(img_bytes, "uint8"), 1)
    # 转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    # 预处理
    img = pre_process_image(img_gray, h, w)
    # 预测
    text = inference(img, h, w)
    text = ''.join(text)
    print("text:{}".format(text))
    return {'text': text}


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    """
    Usage: 
    export KMP_DUPLICATE_LIB_OK=TRUE
    python rest.py -l output/crnn.horizontal.061.pth -v output/crnn.vertical.090.pth -d cuda
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('-l', "--weight-path-horizontal", type=str, default=None, help="weight path")
    parse.add_argument('-v', "--weight-path-vertical", type=str, default=None, help="weight path")
    parse.add_argument('-d', "--device", type=str, default='cpu', help="cpu or cuda")
    args = parse.parse_args(sys.argv[1:])
    alpha = cfg.word.get_all_words()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    # 加载权重，水平方向
    h_net = crnn.CRNN(num_classes=len(alpha))
    h_net.load_state_dict(torch.load(args.weight_path_horizontal, map_location='cpu')['model'])
    h_net.eval()
    h_net.to(device)
    # 垂直方向
    v_net = crnn.CRNNV(num_classes=len(alpha))
    v_net.load_state_dict(torch.load(args.weight_path_vertical, map_location='cpu')['model'])
    v_net.eval()
    v_net.to(device)
    # 启动restful服务
    start_tornado(app, 5000)
