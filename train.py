# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/12/4 下午7:47
 @Author  : yizuotian
 @Description    :
"""

import random
import sys
import cv2
import numpy as np
import torch
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
# import torchvision.transforms as transforms

import crnn
import argparse


def random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


class Generator(Dataset):
    def __init__(self):
        super(Generator, self).__init__()
        self.alpha = ' 0123456789abcdefghijklmnopqrstuvwxyz'
        self.max_len = 20
        self.min_len = 10

    def __getitem__(self, item):
        image = np.random.rand(32, 512, 3) * 255
        # image = np.zeros((32, 512, 3))
        image = image.astype(np.uint8)
        target_len = int(np.random.uniform(self.min_len, self.max_len, size=1))
        indices = np.random.choice(range(1, len(self.alpha)), target_len)
        text = [self.alpha[idx] for idx in indices]
        color = random_color()
        # print(color)
        # cv2.imwrite('a.jpg', image)
        # print(''.join(text))
        image = cv2.putText(image, ''.join(text), (32, 26), cv2.FONT_HERSHEY_PLAIN, 2.3, color,
                            thickness=2)
        # cv2.imwrite('{}.jpg'.format(np.random.randint(10)), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.transpose(image[:, :, np.newaxis], axes=(2, 1, 0))  # [H,W,C]=>[C,W,H]
        image = image.astype(np.float32) / 255.
        image -= 0.5
        image /= 0.5
        target = np.zeros(shape=(self.max_len,), dtype=np.long)
        target[:target_len] = indices
        input_len = 31 - 2
        return image, target, input_len, target_len

    def __len__(self):
        return 10000


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
