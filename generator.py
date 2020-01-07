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

from fontutils import FONT_CHARS_DICT


def random_color(lower_val, upper_val):
    return [random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val)]


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
    def __init__(self, alpha):
        super(Generator, self).__init__()
        # self.alpha = ' 0123456789abcdefghijklmnopqrstuvwxyz'
        self.alpha = alpha
        self.alpha_list = list(alpha)
        self.min_len = 5
        self.max_len_list = [16, 19, 24, 26]
        self.max_len = max(self.max_len_list)
        self.font_size_list = [30, 25, 20, 18]
        self.font_path_list = list(FONT_CHARS_DICT.keys())
        self.font_list = []  # 二位列表[size,font]
        for size in self.font_size_list:
            self.font_list.append([ImageFont.truetype(font_path, size=size)
                                   for font_path in self.font_path_list])

    @classmethod
    def gen_background(cls):
        a = random.random()
        if a < 0.1:
            return np.random.rand(32, 512, 3) * 100
        elif a < 0.8:
            return np.zeros((32, 512, 3)) * np.array(random_color(0, 100)) * 100 / 255
        else:
            b = random.random()
            return b * np.random.rand(32, 512, 3) * 100 + \
                   (1 - b) * np.zeros((32, 512, 3)) * np.array(random_color(0, 100)) * 100 / 255

    def gen_image(self):
        idx = np.random.randint(len(self.max_len_list))
        image = self.gen_background()
        image = image.astype(np.uint8)
        target_len = int(np.random.uniform(self.min_len, self.max_len_list[idx], size=1))

        while True:
            # 随机选择size,font
            size_idx = np.random.randint(len(self.font_size_list))
            font_idx = np.random.randint(len(self.font_path_list))
            font = self.font_list[size_idx][font_idx]
            font_path = self.font_path_list[font_idx]
            # 在选中font字体的可见字符中随机选择target_len个字符
            text = np.random.choice(FONT_CHARS_DICT[font_path], target_len)
            text = ''.join(text)
            w, h = font.getsize(text)
            # 文字在图像尺寸内,即退出
            if w <= 512 or h <= 32:
                break
            print('font_path:{},size:{}'.format(font_path, self.font_size_list[size_idx]))

        # 对应的类别
        indices = np.array([self.alpha.index(c) for c in text])
        # 计算边缘空白大小
        h_margin = max(32 - h, 1)
        w_margin = max(512 - w, 1)

        color = random_color(105, 255)
        image = put_text(image, np.random.randint(w_margin), np.random.randint(h_margin),
                         text, font, tuple(color))
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
        input_len = 31
        return image, target, input_len, target_len

    def __len__(self):
        return len(self.alpha) * 100


def test_image_gen():
    from config import cfg
    gen = Generator(cfg.word.get_all_words())
    for i in range(100):
        im, indices, target_len = gen.gen_image()
        cv2.imwrite('images/examples-{:03d}.jpg'.format(i + 1), im)
        print(''.join([gen.alpha[j] for j in indices]))


def test_gen():
    from data.words import Word
    gen = Generator(Word().get_all_words())
    for x in gen:
        print(x[1])


def test_font_size():
    font = ImageFont.truetype('fonts/simsun.ttc')
    print(font.size)
    font.size = 20
    print(font.size)


if __name__ == '__main__':
    test_image_gen()
    # test_gen()
    # test_font_size()
