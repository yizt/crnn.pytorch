# -*- coding: utf-8 -*-
"""
 @File    : eval.py
 @Time    : 2020/6/22 下午5:05
 @Author  : yizuotian
 @Description    :  使用生成数据评估
"""

import argparse
import itertools
import sys

import numpy as np
import torch

import crnn
from config import cfg
from train import Generator


def inference_single_image(net, image, device=None):
    image = np.expand_dims(image, axis=0)  # 扩展batch维
    image = torch.FloatTensor(image)
    if device:
        image = image.to(device)
    predict = net(image)[0].cpu().detach().numpy()  # [W,num_classes]
    label = np.argmax(predict[:], axis=1)
    label = [k for k, g in itertools.groupby(list(label))]
    label = np.array(label)
    return label[label > 0]  # 去除空格


def main(args):
    alpha = cfg.word.get_all_words()
    if args.direction == 'horizontal':
        net = crnn.CRNN(num_classes=len(alpha))
    else:
        net = crnn.CRNNV(num_classes=len(alpha))

    device = torch.device(
        'cuda:{}'.format(args.local_rank) if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(args.weight_path, map_location='cpu')['model'])
    net.to(device)
    net.eval()
    # load image
    data_set = Generator(cfg.word.get_all_words(), args.direction)

    acc_list = []
    for i in range(args.eval_num):
        image, target, input_len, target_len = data_set[i]
        predict_text = inference_single_image(net, image, device)
        gt = target[:target_len]
        # print("{} {}".format(gt, predict_text))
        acc_list.append(len(gt) == len(predict_text) and np.allclose(gt, predict_text))

    # 精度计算
    acc = np.array(acc_list).mean()
    print('acc:{:.3f}'.format(acc))


if __name__ == '__main__':
    """
    Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE
    python eval.py --weight-path /path/to/chk.pth --direction horizontal \
    --eval-num 1000 --device cpu
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parse.add_argument("--direction", type=str, choices=['horizontal', 'vertical'],
                       default='horizontal', help="horizontal or vertical")
    parse.add_argument("--weight-path", type=str, default=None, help="weight path")
    parse.add_argument("--eval-num", type=int, default=1000, help="number of images to evaluate")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
