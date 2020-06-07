# -*- coding: utf-8 -*-
"""
 @File    : crnn.py
 @Time    : 2019/12/2 下午8:21
 @Author  : yizuotian
 @Description    :
"""

from collections import OrderedDict

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, 64)),  # [B,64,W,32]
            ('max_pool_1', nn.MaxPool2d(2, 2)),  # [B,64,W/2,16]

            ('conv_block_2', _ConvBlock(64, 128)),  # [B,128,W/2,16]
            ('max_pool_2', nn.MaxPool2d(2, 2)),  # [B,128,W/4,8]

            ('conv_block_3_1', _ConvBlock(128, 256)),  # [B,256,W/4,8]
            ('conv_block_3_2', _ConvBlock(256, 256)),  # [B,256,W/4,8]
            ('max_pool_3', nn.MaxPool2d((2, 2), (1, 2))),  # [B,256,W/4,4]

            ('conv_block_4_1', _ConvBlock(256, 512, bn=True)),  # [B,512,W/4,4]
            ('conv_block_4_2', _ConvBlock(512, 512, bn=True)),  # [B,512,W/4,4]
            ('max_pool_4', nn.MaxPool2d((2, 2), (1, 2))),  # [B,512,W/4,2]

            ('conv_block_5', _ConvBlock(512, 512, kernel_size=2, padding=0))  # [B,512,W/4,1]
        ]))

        self.rnn1 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.transcript = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        x = self.cnn(x)  # [B,512,W/16,1]
        x = torch.squeeze(x, 3)  # [B,512,W]
        x = x.permute([0, 2, 1])  # [B,W,512]
        x, h1 = self.rnn1(x)
        x, h2 = self.rnn2(x, h1)
        x = self.transcript(x)
        return x


class CRNNV(nn.Module):
    """
    垂直版CRNN,不同于水平版下采样4倍，下采样16倍
    """

    def __init__(self, num_classes, **kwargs):
        super(CRNNV, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, 64)),  # [B,64,W,32]
            ('max_pool_1', nn.MaxPool2d(2, 2)),  # [B,64,W/2,16]

            ('conv_block_2', _ConvBlock(64, 128)),  # [B,128,W/2,16]
            ('max_pool_2', nn.MaxPool2d(2, 2)),  # [B,128,W/4,8]

            ('conv_block_3_1', _ConvBlock(128, 256)),  # [B,256,W/4,8]
            ('conv_block_3_2', _ConvBlock(256, 256)),  # [B,256,W/4,8]
            ('max_pool_3', nn.MaxPool2d((1, 2), 2)),  # [B,256,W/8,4]

            ('conv_block_4_1', _ConvBlock(256, 512, bn=True)),  # [B,512,W/8,4]
            ('conv_block_4_2', _ConvBlock(512, 512, bn=True)),  # [B,512,W/8,4]
            ('max_pool_4', nn.MaxPool2d((1, 2), 2)),  # [B,512,W/16,2]

            ('conv_block_5', _ConvBlock(512, 512, kernel_size=2, padding=0))  # [B,512,W/4,1]
        ]))

        self.rnn1 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.transcript = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        x = self.cnn(x)  # [B,512,W/16,1]
        x = torch.squeeze(x, 3)  # [B,512,W]
        x = x.permute([0, 2, 1])  # [B,W,512]
        x, h1 = self.rnn1(x)
        x, h2 = self.rnn2(x, h1)
        x = self.transcript(x)
        return x


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=False):
        super(_ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))


if __name__ == '__main__':
    import torchsummary

    net = CRNN(num_classes=1000)
    torchsummary.summary(net, input_size=(1, 512, 32), batch_size=1)
