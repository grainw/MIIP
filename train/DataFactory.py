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
        self.type = type if type is not None else FileType.PANDAS_JSON
        self.dataFrame = PdFileUtils(self.path,self.type).doRead()

    def output_1(self):
         pass

    #归一化处理，去掉标点符号和英文，把所有数字转换为'NUM'
    def normalize(text):
        text = re.compile(ur'[^0-9\u4e00-\u9fa5]').sub('', text)
        return text




class FileUtils:
    __metaclass__ = ABCMeta
    def __init__(self,path,type = None):
        self.path = path
        self.type = type
    @abstractmethod
    def doRead(self):
        '''
        ???
        :return:
        '''

class PdFileUtils(FileUtils):
    def doRead(self):
        print 'pd'+str(self.path)
        if self.type ==FileType.PANDAS_JSON:
            self.doReadJson()
        elif self.type ==FileType.PANDAS_CVS:
            self.doReadCvs()

    def doReadJson(self):
        print 'json'

    def doReadCvs(self):
        print 'cvs'


class MyException(Exception):
    def __init__(self, type):
        Exception.__init__(self)
        self.type = type

class FileType:
    PANDAS_JSON = 0
    PANDAS_CVS = 1
if __name__ == '__main__':
    df = DataFactoryImpl('d://aa',FileType.PANDAS_CVS)
    DF = df.output_1()
    # df = pd.read_json("dxy2.json")
    # # print df.index
    # # print df.columns
    # # print df[df.columns[1:2]]
    # print df.iloc[3:5,0:2]
    #
    # print df.iloc[3:5]




