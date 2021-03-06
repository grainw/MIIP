# coding: utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
import threading
import re


class FileType:
    JSON = 0
    CSV = 1
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
        if self.type == FileType.JSON:
            return self.doReadJson()
        elif self.type == FileType.CSV:
            return self.doReadCsv()
        elif self.type == FileType.TEXT:
            return self.doReadText()

    #Read json files
    def doReadJson(self):
        return pd.read_json(self.path)

    #Read csv files
    def doReadCsv(self):
        return pd.read_csv(self.path, names=self.names)

    #Read text files
    def doReadText(self):
        return pd.read_table(self.path, names=self.names)

    #def write(self):
    #    self.data.to_json(self.path)

class MyException(Exception):
    def __init__(self, type):
        Exception.__init__(self)
        self.type = type
