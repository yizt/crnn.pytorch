# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/12/4 下午7:47
 @Author  : yizuotian
 @Description    :
"""

import argparse
import sys

import numpy as np
import torch
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import crnn
from generator import Generator


# import torchvision.transforms as transforms


def train(args):
    criterion = CTCLoss()
    criterion = criterion.cuda()
    data_set = Generator()
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=3)
    net = crnn.CRNN(len(data_set.alpha))
    if args.init_epoch > 0:
        net.load_state_dict(torch.load('crnn.{:03d}.pth'.format(args.init_epoch)))
    net = net.cuda()
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    for epoch in range(args.init_epoch, args.epochs):
        epoch_loss = 0.0
        for image, target, input_len, target_len in tqdm(data_loader):
            image = image.cuda()
            # print(target, target_len, input_len)
            outputs = net(image.to(torch.float32))  # [B,N,C]
            outputs = torch.log_softmax(outputs, dim=2)
            outputs = outputs.permute([1, 0, 2])  # [N,B,C]
            loss = criterion(outputs[2:], target, input_len, target_len)
            # 梯度更新
            net.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * image.size(0)
            if np.isnan(loss.item()):
                print(target, input_len, target_len)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, args.epochs, epoch_loss / len(data_set)))
        torch.save(net.state_dict(), 'crnn.{:03d}.pth'.format(epoch + 1))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch-size", type=int, default=16, help="batch size")
    parse.add_argument("--epochs", type=int, default=40, help="epochs")
    parse.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parse.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    arguments = parse.parse_args(sys.argv[1:])
    train(arguments)
