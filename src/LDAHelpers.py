#! /usr/bin/python
#coding=utf-8
import pickle
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class LDAHelpers:
    def __init__(self, data, all_words):
        '''
        data is a pandas DataFrame which contains ['_id', 'content', 'url', 'zhuti']
        content is a tokenized document.
        all_words can be a simple list which contains all the words for all the documents
        stop_words is also a simple list
        '''
        self.data = data
        self.all_words = all_words

    def getTFMat(self):
        '''
        Return a mat whcih contains TF to per-document.
        '''
        mat = []
        for idx, series in self.data.iterrows():
            arr = []
            for word in self.all_words:
                arr.append(series["content"].count(word))
            mat.append(arr)
        return mat
