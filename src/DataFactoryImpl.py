# coding: utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re


#归一化处理，去掉标点符号和英文，把所有数字转换为'NUM'
def normalize(text):
    text = re.compile(ur'[^0-9\u4e00-\u9fa5]').sub('', text)
    return text

class DataFactoryImpl:
    '''
    Class for dealing with data.
    '''
    def __init__(self, data, user_dict):
        self.data = data
        self.user_dict = user_dict
        self.n_samples = len(self.data)
    def output(self):
         pass

    def getAllWords():
        '''
        对文章做分词, 接受pandas的dataFrame, 处理返回一个pandas的dataFrame,包含主题和分好词之后的所有词语
        dataFrame 的数据结构为[_id, content, url, zhuti]
        '''
        element = []
        jieba.load_userdict(self.user_dict)
        for i in range(self.n_samples):
            seg_list = pseg.cut(self.data["content"][i])
            words = [normalize(word) for word,flag in seg_list if normalize(word) != '' and normalize(word) not in stop_words_list and flag == 'n']
            list_words.extend((self.data["zhuti"][i], words))
        return pd.DataFrame(list_words)
