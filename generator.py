# -*- coding: utf-8 -*-
"""
 @File    : generator.py
 @Time    : 2019/12/22 下午8:22
 @Author  : yizuotian
 @Description    :  中文数据生成器
"""
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data.dataset import Dataset

from data.words import Word


def random_color():
    return [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]


def put_text(image, x, y, text, font, color=None):
    """
    写中文字
    :param image:
    :param x:
    :param y:
    :param text:
    :param font:
    :param color:
    :return:
    """

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    color = (255, 0, 0) if color is None else color
    draw.text((x, y), text, color, font=font)
    return np.array(im)


class Generator(Dataset):
    def __init__(self):
        super(Generator, self).__init__()
        # self.alpha = ' 0123456789abcdefghijklmnopqrstuvwxyz'
        self.alpha = Word().get_all_words()
        self.min_len = 5
        self.max_len_list = [16, 19, 24, 26]
        self.max_len = max(self.max_len_list)
        self.font_size_list = [30, 25, 20, 18]
        self.font_list = [ImageFont.truetype('fonts/simsun.ttc', size=s) for s in self.font_size_list]

    def gen_image(self):
        idx = np.random.randint(len(self.max_len_list))
        image = np.random.rand(32, 512, 3) * 100
        image = image.astype(np.uint8)
        target_len = int(np.random.uniform(self.min_len, self.max_len_list[idx], size=1))
        indices = np.random.choice(range(1, len(self.alpha)), target_len)
        text = [self.alpha[idx] for idx in indices]
        color = random_color()
        image = put_text(image, 32, np.random.randint(1, 32 - self.font_size_list[idx]), ''.join(text),
                         self.font_list[idx], tuple(color))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if random.random() > 0.5:
            image = 255 - image
        return image, indices, target_len

    def __getitem__(self, item):
        image, indices, target_len = self.gen_image()
        image = np.transpose(image[:, :, np.newaxis], axes=(2, 1, 0))  # [H,W,C]=>[C,W,H]
        image = image.astype(np.float32) / 255.
        image -= 0.5
        image /= 0.5
        target = np.zeros(shape=(self.max_len,), dtype=np.long)
        target[:target_len] = indices
        input_len = 31 - 2
        return image, target, input_len, target_len

    def __len__(self):
        return len(self.alpha) * 100


def test_image_gen():
    gen = Generator()
    im, indices, target_len = gen.gen_image()
    cv2.imwrite('examples.jpg', im)
    print(''.join([gen.alpha[i] for i in indices]))


if __name__ == '__main__':
    test_image_gen()
