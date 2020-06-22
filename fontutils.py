# -*- coding: utf-8 -*-
"""
 @File    : fontutils.py
 @Time    : 2020/1/7 上午11:53
 @Author  : yizuotian
 @Description    :
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from config import cfg

word_set = set(cfg.word.get_all_words())


def get_all_font_chars():
    """

    :return: font_chars_dict: dict{font_path:[visible chars]
    """
    font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_path_list = [os.path.join(font_dir, font_name)
                      for font_name in os.listdir(font_dir)]
    font_list = [ImageFont.truetype(font_path, size=10) for font_path in font_path_list]
    font_chars_dict = dict()
    for font, font_path in zip(font_list, font_path_list):
        font_chars = get_font_chars(font_path)
        font_chars = [c.strip() for c in font_chars if len(c) == 1 and
                      word_set.__contains__(c) and
                      is_char_visible(font, c)]  # 可见字符
        font_chars = list(set(font_chars))  # 去重
        font_chars.sort()
        font_chars_dict[font_path] = font_chars

    return font_chars_dict


def to_unicode(glyph):
    return json.loads(f'"{glyph}"')


def get_font_chars(font_path):
    font = TTFont(font_path, fontNumber=0)
    glyph_names = font.getGlyphNames()
    char_list = []
    for idx, glyph in enumerate(glyph_names):
        if glyph[0] == '.':  # 跳过'.notdef', '.null'
            continue
        if glyph == 'union':
            continue
        if glyph[:3] == 'uni':
            glyph = glyph.replace('uni', '\\u')
        if glyph[:2] == 'uF':
            glyph = glyph.replace('uF', '\\u')
        if glyph == '\\uversal':
            continue

        char = to_unicode(glyph)
        char_list.append(char)
    return char_list


def is_char_visible(font, char):
    """
    是否可见字符
    :param font:
    :param char:
    :return:
    """
    gray = Image.fromarray(np.zeros((20, 20), dtype=np.uint8))
    draw = ImageDraw.Draw(gray)
    draw.text((0, 0), char, 100, font=font)
    visible = np.max(np.array(gray)) > 0
    return visible


FONT_CHARS_DICT = get_all_font_chars()

if __name__ == '__main__':
    print(FONT_CHARS_DICT)
