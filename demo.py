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
from config import cfg


def main(args):
    alpha = cfg.word.get_all_words()
    net = crnn.CRNN(num_classes=len(alpha))
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'])
    net.eval()

    image = Image.open(args.image_path).convert('L')
    image = np.array(image).T  # [W,H]
    image = image.astype(np.float32) / 255.
    image -= 0.5
    image /= 0.5
    image = image[np.newaxis, np.newaxis, :, :]
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
