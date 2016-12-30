#!/usr/bin/python
# -*- coding: UTF-8 -*-

from functions import *
import numpy as np
import threading
import time
import sys
import os
sys.path.append(os.path)
from CommonUtils import *


exitFlag = 0
# log = src.LoggerFacotory.Logger('test.log', logging.INFO, logging.INFO)
class LdaThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, data_sample,data_label,stop_words_list,theta):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data_sample = data_sample
        self.data_label  = data_label
        self.n_features = 2000
        self.stop_words_list = stop_words_list
        self.theta = theta

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        log.info( "Starting " + self.name+","+str(self.theta))
        log.info( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        generateWords(self.threadID,self.data_sample,self.data_label,self.n_features,self.stop_words_list,self.theta)
        log.info( "ending" + self.name+"....")
        log.info( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def generateWords(threadID,data_sample,data_label, n_features, stop_words_list, theta):
    for i in range(len(data_sample)):
        data = data_sample[i]
        allDocWords = []
        allDocWords.append(data_label[i] + "#")
        log.info("第"+str(threadID)+"批次第"+str(i)+"偏文章做lda...........")
        allWords = getAllWords([data], stop_words_list)
        log.info("文件写入所有词")
        allDocWords.extend(allWords)
        writeAllWords(allDocWords)
        # lda主题词提取
        X = getWordsTFMat([data],list(set(allWords)) , stop_words_list)
        model = lda.LDA(1, n_iter=1000, random_state=1)
        model.fit(np.array(X))
        dictWords = {}
        for j in theta:
            words = []
            w = 0.0
            words.append(data_label[i]+"#")
            for k in model.topic_word_[0].argsort()[:-len(model.topic_word_[0])- 1:-1]:
                if w <= j:
                    w += model.topic_word_[0][k]
                    words.append(allWords[k])
                else:
                    break
            if j in dictWords.keys():
                dictWords[j].append(words)
            else :
                dictWords[j] = []
                dictWords[j].append(words)
        log.info("文件写入权重词")
        writeWords(dictWords,theta)


def writeWords(words,theta):
     for i in theta:
        file = open('wordset'+str(i), 'a')
        file.writelines(["%s\n" % ' '.join(item) for item in words[i]])
        file.close()

def writeAllWords(allWords):
     # 词频统计
     # file = open('allwords', 'a')
     file = open('allwords1', 'a')
     # file.writelines(["%s\n" % ' '.join(item)  for item in allWords])
     file.writelines(["%s\n" % ' '.join(allWords)])
     file.close()

def writeStopWords(words,allWords,theta):
    for i in theta:
        a = set(list(np.array(allWords).ravel()))-set(list(np.array(words[i]).ravel()))
        file = open('stopwords_theta'+str(i), 'a')
        file.writelines(["%s\n" % ' '.join(item) for item in a])
        file.close()