# coding: utf-8
from abc import ABCMeta, abstractmethod
from CommonUtils import *
from FileUtils import *
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re
import jieba.posseg as pseg
import jieba
import jieba.analyse

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
            words = [normalize(word) for word, flag in seg_list if normalize(word) != '' and normalize(word) not in self.stop_words and flag in ['n','ns','nsf','nt','nz','nl','ng'] ]
            item.append((series['_id'], words, series['from'], series['url'], series['zhuti']))
        return pd.DataFrame(item, columns=['_id', 'content', 'from', 'url', 'zhuti'])



if __name__ == '__main__':
    jieba.load_userdict(userDictPath)
    stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
    content = '裂腭####吞咽困难####牙关紧闭####咽痛####吞咽痛####张口困难####听力减退####食物返流至鼻腔####说话带鼻音####耳闷####听力下降####'
    content1 = '兔唇####巨口####吞咽困难####嘴角降肌发育不良####唇炎####糜烂型唇炎####干燥脱屑型唇####腺型唇炎####肉芽肿性唇炎####下唇微痒增大如蚕茧硬结####口唇皲裂####囊肿####唇外翻####米舍尔肉芽肿性唇炎####口唇渗液####唇上长有小水泡####结节####溃疡####唇干裂####唇干裂####口唇皲裂####口唇渗液####唇水疱####上唇肥厚'
    content2 = '慢性边缘剥脱性舌炎####游走性舌炎####舌炎####舌岩####舌疳####吞咽困难####舌炎####舌头上有长时间不愈溃疡####舌痛####进食困难####舌病####地图舌####毛舌####正中菱舌####恶心与呕吐####沟裂舌####沟舌####沟纹舌####裂缝舌####裂纹舌####阴囊舌####皱襞舌####溃疡####刺痛####黑毛舌####毛舌####舌背中后部黑色丛毛####口干####口苦####黑棘底红舌####黑毛舌####黑毛舌病####黑舌病####舌黑变病########唇舌水肿及面瘫综合症####味觉减退####听觉过敏####疼痛####面神经瘫痪####舌异位甲状腺####咽部异物感####咽痛####吞咽不利####呼吸异常####呼吸困难####'
    seg_list = pseg.cut(content)
    words = [normalize(word) for word, flag in seg_list if normalize(word) != '' and normalize(word) not in stop_words and flag in ['n','ns','nsf','nt','nz','nl','ng'] ]
    s =  ','.join(words)
    tags = jieba.analyse.extract_tags(s,topK=100,withWeight=True)
    for tag in tags:
        print tag[0]+" "+str(tag[1])


