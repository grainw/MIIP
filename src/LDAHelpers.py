#! /usr/bin/python
#coding=utf-8
import pickle
import glob
import numpy as np
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class LDAHelpers:
    def __init__(self, data, all_words,slabel=None,avgValue = None):
        '''
        data is a pandas DataFrame which contains ['_id', 'content', 'url', 'zhuti']
        content is a tokenized document.
        all_words can be a simple list which contains all the words for all the documents
        stop_words is also a simple list
        '''
        self.data = data
        self.all_words = all_words
        self.slabel = slabel
        self.avgValue = avgValue


    def getTFMat(self):
        '''
        Return a mat whcih contains TF to per-document.
        '''
        mat = []
        for idx, series in self.data.iterrows():
            arr = []
            for word in self.all_words:
                if self.slabel !=None and len(self.slabel)>0:
                    if word in series['fetures']:
                        arr.append(self.avgValue)
                    else:
                        arr.append(series["content"].count(word))
                else:
                    # 统计词频
                    arr.append(series["content"].count(word))
                    # tf = float(series["content"].count(word))/len(series["content"])
                    # num = 0
                    # for k,m in self.data.iterrows():
                    #     if m["content"].count(word)>0:
                    #         num = num + 1
                    # idf = math.log10(float(len(self.data))/(num+1))
                    # arr.append(tf*idf)
                    # print word +" :"+str(tf*idf)
            mat.append(arr)
        return mat