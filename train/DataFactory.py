# coding: utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re


class DataFactoryImpl:

    instance=None
    def __init__(self,path,type=None):
        self.path = path
        self.type = type if type is not None else FileType.PANDAS_TYPE
        rs = self.input_1()
        self.rs = rs

    def input_1(self):
        RS = {
            FileType.DEFALUT_TYPE: DefaultFileUtils(self.path),
            FileType.JSON_TYPE: JsonFileUtils(self.path),
            FileType.NUMPY_TYPE: NpFileUtils(self.path),
            FileType.PANDAS_TYPE: PdFileUtils(self.path)
         }

        return RS[self.type]

    def output_1(self):
         self.rs.doRead()



class DataElement:
    def __init__(self,content,label,url):
        self.content = content
        self.label = label
        self.url = url

class FileUtils:
    __metaclass__ = ABCMeta
    def __init__(self,path):
        self.path = path
    @abstractmethod
    def doRead(self):
        '''
        ???
        :return:
        '''
    #归一化处理，去掉标点符号和英文，把所有数字转换为'NUM'
    def normalize(text):
        text = re.compile(ur'[^0-9\u4e00-\u9fa5]').sub('', text)
        return text

class JsonFileUtils(FileUtils):

    def doRead(self):
        print 'json'+str(self.path)

class DefaultFileUtils(FileUtils):

    def doRead(self):
        # list = []
        # for line in open(self.path, 'r'):
        #     word = line.strip().decode('utf-8')
        #     super.normalize(word)
        #     de = DataElement()
        #     de.content = ' '.join(super.normalize(word))
        #     list.append(de)
        print 'defulat'+str(self.path)
        # return list

class NpFileUtils(FileUtils):

    def doRead(self):
        # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
        # data = np.loadtxt(self.path, dtype=float, delimiter=',', converters={4: iris_type})
        print 'np'+str(self.path)
    # def iris_type(s):
    #     it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    #     return it[s]

class PdFileUtils(FileUtils):

    def doRead(self):
        print 'pd'+str(self.path)

class MyException(Exception):
    def __init__(self, type):
        Exception.__init__(self)
        self.type = type

class FileType:
    DEFALUT_TYPE = 0
    JSON_TYPE = 1
    NUMPY_TYPE = 2
    PANDAS_TYPE = 3
if __name__ == '__main__':
    df = DataFactoryImpl('d://aa')
    DF = df.output_1()
    df = pd.read_json("dxy2.json")
    print df["_id"]


