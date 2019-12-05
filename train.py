# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/12/4 下午7:47
 @Author  : yizuotian
 @Description    :
"""

import random

import cv2
import numpy as np
import torch
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import crnn


def random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


class Generator(Dataset):
    def __init__(self):
        super(Generator, self).__init__()
        self.alpha = ' 0123456789abcdefghijklmnopqrstuvwxyz'
        self.max_len = 20
        self.min_len = 2

    def __getitem__(self, item):
        image = np.random.rand(32, 512, 3) * 255
        image = image.astype(np.uint8)
        target_len = int(np.random.uniform(self.min_len, self.max_len, size=1))
        indices = np.random.choice(len(self.alpha), target_len)
        text = [self.alpha[idx] for idx in indices]
        color = random_color()
        # print(color)
        # cv2.imwrite('a.jpg', image)
        # print(''.join(text))
        image = cv2.putText(image, ''.join(text), (2, 26), cv2.FONT_HERSHEY_PLAIN, 2.3, color,
                            thickness=2)
        cv2.imwrite('a.jpg', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.transpose(image[:, :, np.newaxis], axes=(2, 1, 0))  # [H,W,C]=>[C,W,H]
        target = np.zeros(shape=(self.max_len,), dtype=np.long)
        target[:target_len] = indices
        input_len = 31
        return image, target, input_len, target_len

    def __len__(self):
        return 10000


def train():
    criterion = CTCLoss()
    data_set = Generator()
    data_loader = DataLoader(data_set, batch_size=16, shuffle=True,
                             num_workers=3)
    net = crnn.CRNN(len(data_set.alpha))
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=1e-3,
                           betas=(0.9, 0.999))

    for epoch in range(10):
        for image, target, input_len, target_len in tqdm(data_loader):
            # print(target, target_len, input_len)
            outputs = net(image.to(torch.float32))  # [B,N,C]
            outputs = torch.log_softmax(outputs, dim=2)
            outputs = outputs.permute([1, 0, 2])
            loss = criterion(outputs, target, input_len, target_len)
            net.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict())


if __name__ == '__main__':
    train()
