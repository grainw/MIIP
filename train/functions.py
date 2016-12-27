# coding: utf-8
from time import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import jieba.posseg as pseg
import pickle
import jieba
import math
#enable parallel for jieba
# jieba.enable_parallel()

import re
import numpy as np
import lda
from numpy import linalg as la


def getTF(n_features, stop_words_list, list_words):
    tf_vectorizer = CountVectorizer(max_features=n_features, stop_words=stop_words_list)
    tf = tf_vectorizer.fit_transform(list_words)
    return tf,tf_vectorizer

def computeTF(allWords):
    tf = {}
    for word in allWords:
        tf[word] = allWords.count(word)
    return tf

def computeDFIDF(data_samples, allWords):
    df = {}
    idf = {}
    for doc in data_samples:
        for word in allWords:
            if doc.find(word) != -1:
                if word in df:
                    df[word] += 1
                else:
                   df[word] = 1
    for item in df:
        idf[item] = 1 + math.log((len(allWords) + 1) / (df[item] + 1))
    return df, idf



def getAllWords(data_samples, stop_words_list):
    list_words = []
    jieba.load_userdict('user.dict')
    for data in data_samples:
        seg_list = pseg.cut(data)
        words = [normalize(word) for word,flag in seg_list if normalize(word) != '' and normalize(word) not in stop_words_list and  flag == 'n']
        list_words.extend(words)
    return list_words

def preprocessor(data_samples, n_features, stop_words_list):
    words = []
    for data in data_samples:
        allWords = list(set(getAllWords([data], stop_words_list)))
        X = getWordsTFMat([data], allWords, stop_words_list)
        model = lda.LDA(1, n_iter=1500, random_state=1)
        model.fit(np.array(X))
        w = 0
        for i in model.topic_word_[0].argsort():
            if w < 0.5:
                w += model.topic_word_[0][i]
                words.append(allWords[i])
    return list(set(words))

def getWordsTFMat(data_samples, allWords, stop_words_list):
    X = []
    jieba.load_userdict('user.dict')
    for data in data_samples:
        seg_list = pseg.cut(data)
        words = [normalize(word) for word,flag in seg_list if normalize(word) != '' and normalize(word) not in stop_words_list and  flag=='n']
        arr = []
        for word in allWords:
            arr.append(words.count(word))
        X.append(arr)
    return X

#归一化处理，去掉标点符号和英文，把所有数字转换为'NUM'
def normalize(text):
    text = re.compile(ur'[^0-9\u4e00-\u9fa5]').sub('', text)
    return text


def printTopWords(model, allWords, n_top_words):
    for topic_idx, topic in enumerate(model.topic_word_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([allWords[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def loadStopWords(infile):
    stopWords = []
    for line in open(infile, 'r'):
        word = line.strip().decode('utf-8')
        stopWords.append(normalize(word))
    return stopWords


def getDataSamples(file):
    json_obj = json.load(open(file))
    items = []
    label = []
    #url   = []
    #load dxy
    for item in json_obj:
        if 'content' in item:
            items.append(item['content'])
            #print type(item['content'])
            label.append(item['zhuti'])
        if 'jianjia' in item:
            items.append(item['jianjia'] + item['decrible'])
            label.append(item['zhiti'])
            words = [normalize(word) for word in item['other'].split("|") if normalize(word) != '' and normalize(word)]
            for i in words:
                label.append(i)
        #url.append(item['url'])
    #load 39.net
    return items, label #url

def getTestTopWords(distr, words_list, n_top_words):
    new_distr = []
    words = []
    for topic in distr:
        idx = topic.argsort()[:-n_top_words -1:-1]
        new_distr.append(topic[idx])
        words.extend(np.array(words_list)[idx])
    return new_distr, words

def getTrainTopWordsProb(topic_word_distr, test_top_words, wordset):
    topics = []
    for topic in topic_word_distr:
        num = len(test_top_words)
        tmp = []
        for i in range(num):
            if test_top_words[i] in wordset:
                tmp.append(topic[wordset.index(test_top_words[i])])
            else:
                tmp.append(0)
        topics.append(tmp)
    return topics
def normilizeVector(vec):
    print "max: ", np.array(vec).max(), " min: ", np.array(vec).min()

    return list((np.array(vec)-np.array(vec).min()) / (np.array(vec).max() - np.array(vec).min()))

def getMostLikelyTopic(train, test):
    dist = []
    #train = np.array(train)
    #test = np.array(test)
    #normilize vector using sigmoid function
    #norm_test = 1.0 / (1 + np.exp(-test))
    #test = normilizeVector(test)

    for tr in train:
        #tr = normilizeVector(tr)
        #norm_train = 1.0 / (1 + np.exp(-tr))
        #dist.append(test * train / la.norm(test) / la.norm(train))
        #print tr
        dist.append(la.norm(np.array(tr)-np.array(test)))
        #dist.append(sum(tr))
        #dist.append(np.array(tr)*np.array(test))
    #return [np.array(dist).argmax()]
    return np.array(dist).argsort()[0:-5-1:-1]

def getMostRelatedDocs(doc_topic_distr, topicIdx, title):
    idx = []
    for i in topicIdx:
        #idx.extend(doc_topic_distr[i].argsort()[0:-2-1:-1])
        idx.extend(list(doc_topic_distr[:,i].argsort())[-1])#[0:-2-1:-1]
        #print idx
    for i in idx:
        print title[i]
    return idx
