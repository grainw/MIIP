# coding: utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re
import jieba.posseg as pseg
import jieba

#归一化处理，去掉标点符号和英文，把所有数字转换为'NUM'
def normalize(text):
    text = re.compile(ur'[^0-9\u4e00-\u9fa5]').sub('', text)
    return text

class DataFactoryImpl:
    '''
    Class for dealing with data.
    '''
    def __init__(self, data, stop_words, user_dict=None):
        self.data = data
        self.user_dict = user_dict
        self.stop_words = list(stop_words[stop_words.columns[0]])
    def output(self):
         pass

    def splitString(self):
        words = []
        for idx, series in self.data.iterrows():
            string = series['content'].split(' ')
            words.extend(string)
        return list(set(words))

    def getAllWords(self):
        '''
        对文章做分词, 接受pandas的dataFrame, 处理返回一个pandas的dataFrame,包含主题和分好词之后的所有词语
        dataFrame 的数据结构为[_id, content, from, url, zhuti]
        '''
        item = []
        if self.user_dict:
            jieba.load_userdict(self.user_dict)
        for idx, series in self.data.iterrows():
            seg_list = pseg.cut(series['content'])
            words = [normalize(word) for word, flag in seg_list if normalize(word) != '' and normalize(word) not in self.stop_words and flag == 'n']
            item.append((series['_id'], words, series['from'], series['url'], series['zhuti']))
        return pd.DataFrame(item, columns=['_id', 'content', 'from', 'url', 'zhuti'])
