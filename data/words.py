# -*- coding: utf-8 -*-
"""
 @File    : words.py
 @Time    : 2019/12/22 下午8:48
 @Author  : yizuotian
 @Description    :
"""
import codecs
import os


class Word(object):
    def __init__(self,
                 chinese_word=True,
                 alphabet=True,
                 digit=True,
                 punctuation=True,
                 currency=True
                 ):
        """

        :param chinese_word: 中文字
        :param alphabet: 英文字母
        :param digit: 数字
        :param punctuation: 标点符号
        :param currency: 货币符号
        """
        self.chinese_word = chinese_word
        self.alphabet = alphabet
        self.digit = digit
        self.punctuation = punctuation
        self.currency = currency

    @classmethod
    def get_digits(cls):
        return '0123456789'

    @classmethod
    def get_alphabet(cls):
        return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    @classmethod
    def get_chinese_words(cls):
        cur_dir = os.path.dirname(__file__)
        # f = codecs.open(os.path.join(cur_dir, 'chinese_word.txt'),
        #                 mode='r', encoding='utf-8')
        f = codecs.open(os.path.join(cur_dir, 'char_std_5990.txt'),
                        mode='r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        lines = [l.strip() for l in lines]
        return ''.join(lines)

    @classmethod
    def get_punctuations(cls):
        return "。，、；：？！…-·ˉˇ¨‘'“”～‖∶＂＇｀｜〃〔〕〈〉《》「」『』．.〖〗【】（）［］｛｝"

    @classmethod
    def get_currency(cls):
        return '$¥'

    def get_all_words(self):
        # words = ' '
        # if self.chinese_word:
        #     words += self.get_chinese_words()
        # if self.alphabet:
        #     words += self.get_alphabet()
        # if self.digit:
        #     words += self.get_digits()
        # if self.punctuation:
        #     words += self.get_punctuations()
        # if self.currency:
        #     words += self.get_currency()
        # return words
        cur_dir = os.path.dirname(__file__)
        f = codecs.open(os.path.join(cur_dir, 'all_words.txt'),
                        mode='r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        lines = [l.strip() for l in lines]
        return ' '+''.join(lines)


if __name__ == "__main__":
    w = Word()
    print(len(w.get_all_words()) == len(set(w.get_all_words())))
    print(w.get_chinese_words())
    print(w.get_all_words())
    print(w.get_all_words().__contains__(' '))
