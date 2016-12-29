# coding: utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re


class FileType:
    PANDAS_JSON = 0
    PANDAS_CVS = 1
    TEXT = 2

class FileUtils:
    '''
    Class for IO
    '''
    def __init__(self, path, fileType, names=None):
        self.path = path
        self.type = fileType
        self.names = names
    def doRead(self):
        if self.fileType ==FileType.PANDAS_JSON:
            return self.doReadJson()
        elif self.fileType ==FileType.PANDAS_CVS:
            return self.doReadCvs()
        elif self.fileType == TEXT:
            return self.doReadText()

    #Read json files
    def doReadJson(self):
        return pd.read_json(self.path, names=names)

    #Read csv files
    def doReadCsv(self):
        return pd.read_cvs(self.path, names=names)

    #Read text files
    def doReadText(self):
        return pd.read_table(self.path, names=names)


class MyException(Exception):
    def __init__(self, type):
        Exception.__init__(self)
        self.type = type
