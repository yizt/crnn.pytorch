# -*- coding: utf-8 -*-
"""
 @File    : demo.py
 @Time    : 2019-12-5 15:02
 @Author  : yizuotian
 @Description    :
"""
import sys
import argparse
import itertools
import crnn
from PIL import Image
import numpy as np
import torch
import cv2
from config import cfg


def load_image(image_path):
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]
    if h != 32:
        new_w = int(w * 32 / h)
        image = cv2.resize(image, (new_w, 32))

    image = Image.fromarray(image).convert('L')
    # cv2.imwrite(image_path, np.array(image))
    image = np.array(image).T  # [W,H]
    image = image.astype(np.float32) / 255.
    image -= 0.5
    image /= 0.5
    image = image[np.newaxis, np.newaxis, :, :]  # [B,C,W,H]
    return image


def main(args):
    alpha = cfg.word.get_all_words()
    net = crnn.CRNN(num_classes=len(alpha))
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'])
    net.eval()
    # load image
    image = load_image(args.image_path)
    image = torch.FloatTensor(image)

    predict = net(image)[0].detach().numpy()  # [W,num_classes]
    label = np.argmax(predict[:], axis=1)
    label = [alpha[class_id] for class_id in label]
    print(''.join(label))
    label = [k for k, g in itertools.groupby(list(label))]
    print(''.join(label))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image-path", type=str, default=None, help="test image path")
    parse.add_argument("--weight-path", type=str, default=None, help="weight path")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
