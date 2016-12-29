#! /usr/bin/python
#coding=utf-8
import pickle
from functions import *
from LoggerFacotory import *
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class LDAHelpers:
    def __init__(self, data):
        '''
        data is a pandas DataFrame which contains ['_id', 'content', 'url', 'zhuti']
        content is a tokenized document.
        all_words can be a simple list which contains all the words for all the documents
        stop_words is also a simple list
        '''
        self.data = data
        self.n_samples = len(self.data)
        self.all_words = all_words
        self.stop_words = stop_words


    def getTFMat():
        '''
        Return a mat whcih contains TF to per-document.
        '''
        mat = []
        for i in ranage(self.n_samples):
            arr = []
            for word in self.all_words:
                arr.append(self.data["content"][i].count(word))
            mat.append(arr)
        return mat
