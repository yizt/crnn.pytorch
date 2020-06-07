# -*- coding: utf-8 -*-
"""
 @File    : demo.py
 @Time    : 2019-12-5 15:02
 @Author  : yizuotian
 @Description    :
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import crnn
from config import cfg


def load_image(image_path):
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]
    if h != 32 and h < w:
        new_w = int(w * 32 / h)
        image = cv2.resize(image, (new_w, 32))
    if w != 32 and w < h:
        new_h = int(h * 32 / w)
        image = cv2.resize(image, (32, new_h))

    image = Image.fromarray(image).convert('L')
    # cv2.imwrite(image_path, np.array(image))
    image = np.array(image)
    if h < w:
        image = np.array(image).T  # [W,H]
    image = image.astype(np.float32) / 255.
    image -= 0.5
    image /= 0.5
    image = image[np.newaxis, np.newaxis, :, :]  # [B,C,W,H]
    return image


def inference_image(net, alpha, image_path):
    image = load_image(image_path)
    image = torch.FloatTensor(image)

    predict = net(image)[0].detach().numpy()  # [W,num_classes]
    label = np.argmax(predict[:], axis=1)
    label = [alpha[class_id] for class_id in label]
    label = [k for k, g in itertools.groupby(list(label))]
    label = ''.join(label).replace(' ', '')
    return label


def main(args):
    alpha = cfg.word.get_all_words()
    if args.direction == 'horizontal':
        net = crnn.CRNN(num_classes=len(alpha))
    else:
        net = crnn.CRNNV(num_classes=len(alpha))
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'])
    net.eval()
    # load image
    if args.image_dir:
        image_path_list = [os.path.join(args.image_dir, n) for n in os.listdir(args.image_dir)]
        image_path_list.sort()
        for image_path in image_path_list:
            label = inference_image(net, alpha, image_path)
            print("image_path:{},label:{}".format(image_path, label))
    else:
        label = inference_image(net, alpha, args.image_path)
        print("image_path:{},label:{}".format(args.image_path, label))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--direction", type=str, choices=['horizontal', 'vertical'],
                       default='horizontal', help="horizontal or vertical")
    parse.add_argument("--image-path", type=str, default=None, help="test image path")
    parse.add_argument("--weight-path", type=str, default=None, help="weight path")
    parse.add_argument("--image-dir", type=str, default=None, help="test image directory")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
