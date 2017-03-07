# coding: utf-8

from sklearn.cluster import KMeans
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np
from CommonUtils import *
import pickle
from FileUtils import *
from DataFactoryImpl import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from LDAHelpers import *
import lda
import math
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class SkmeansCluster:
    def __init__(self):
        self.fromzhuti = FileUtils(fromFilePath, FileType.CSV).doRead()['zhuti']
        self.data = FileUtils(fromFilePath, FileType.CSV).doRead()
        self.stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
        self.allWords = DataFactoryImpl(self.data, self.stop_words,userDictPath).splitString()
        self.X = LDAHelpers(self.data, self.allWords).getTFMat()
        # trainModel  = pickle.load(open(modelPath))
        # X = trainModel.doc_topic_

    def kmeansCluster(self,clusters):
        kmeans = KMeans(n_clusters=clusters,max_iter=1000,tol=1e-7, random_state=0).fit(self.X)
        list = []
        for i in range(clusters):
            list.append(np.where(kmeans.labels_==i))
        self.printClusterIndexToLabel(list)
        return list

    def printClusterIndexToLabel(self,list,x_label):
        print list
        for j in list:
            arr = []
            for k in j[0]:
                arr.append(x_label[k])
                # arr.append(fromzhuti[trainModel.doc_topic_[:,k].argmax()])
            print ",".join(arr)
    # t = 0.77 口腔
    # t = 0.55 鼻子
    # t =  0.8 ear
    #   t = 0.77 face
    # lunao = t = 0.79
    # yan = t = 0.79
    def hierarchicalCluster(self,x_train,x_label,t = 0.86,flag = False):
        disMat = sch.distance.pdist(x_train,'cosine')
        Z=sch.linkage(disMat,method='average')
        P=sch.dendrogram(Z)
        plt.savefig('plot_dendrogram.png')
        #根据linkage matrix Z得到聚类结果:
        cluster= sch.fcluster(Z,t,criterion='distance',depth=6)
        # print "Original cluster by hierarchy clustering:\n",cluster
        # print max(cluster)
        #
        # print '层次聚类的索引结果...................'
        list = []
        for i1 in range(1,(max(cluster)+1)):
            list.append(np.where(cluster==i1))
        self.printClusterIndexToLabel(list,x_label)
        if flag:
            save_path = os.getcwd() +'\\cluster-'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            os.mkdir(save_path)
            saveModel = open(save_path + '\\cluster_model', 'w')
            pickle.dump(list, saveModel)
        return list


    def printTopWords(self,model, allWords, n_top_words,x_label):

        x_label.index = range(len(x_label.index))
        for topicIdx, topic in enumerate(model.topic_word_):
            print("Topic #%d:" % topicIdx)
            print(" ".join([allWords[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            docIdx = model.doc_topic_[:, topicIdx].argmax()
            # topic 对应的主题和id
            print "Doc: ",  x_label['zhuti'][docIdx], ' ', x_label['_id'][docIdx]

    def getDocTopic(self,x_data,skc,num):
        a = DataFactoryImpl(x_data, skc.stop_words,userDictPath).splitString()
        x_matrix= LDAHelpers(x_data, a).getTFMat()
        model = lda.LDA(num, n_iter = 1500, random_state=1)
        model.fit(np.array(x_matrix))
        skc.printTopWords(model, a, 70,x_label=x_data)

    def getDataSamples(self,file):
        json_obj = json.load(open(file))
        items = []
        for item in json_obj:
            if 'content' in item:
                item['content'].split('')
                items.append(item['content'])
        return items


if __name__ == '__main__':
    skc = SkmeansCluster()
    list = skc.hierarchicalCluster(skc.X,skc.fromzhuti)
    # 大的分类采用层次分类
    # for sigle in list:
    #     print '---------->'
    #     if len(sigle[0])<=2:
    #         continue
    #     x_train = [skc.X[i] for i in sigle[0]]
    #     x_lable = [skc.fromzhuti[i] for i in sigle[0]]
    #     skc.hierarchicalCluster(x_train,t = 0.46,x_label=x_lable)
    # # x_train = [skc.X[i] for i in list[len(list)-1][0]]
    # # x_lable = [skc.fromzhuti[i] for i in list[len(list)-1][0]]
    # # listLast = skc.hierarchicalCluster(x_train,t = 0.65,x_label=x_lable)
    # # for last in listLast:
    # #     x_train = [x_train[i] for i in last[0]]
    # #     x_lable = [x_lable[i] for i in last[0]]
    # #
    # #     # a = DataFactoryImpl(x_train, skc.stop_words,userDictPath).splitString()
    # #     # x_matrix= LDAHelpers(x_train, a).getTFMat()
    # #     # model = lda.LDA(1, n_iter = 1500, random_state=1)
    # #     # model.fit(np.array(x_matrix))
    # #     # skc.printTopWords(model, a, 20,x_label=x_data)
    #
    #
    #
    # # --------------------------------------------------------------
    # x_data =skc.data.ix[list[len(list)-1][0]]
    # # skc.getDocTopic(x_data,skc,len(list[len(list)-1][0]))
    # # +++++++++++++++++++++++++++++++++++++++++++++++
    # a = DataFactoryImpl(x_data, skc.stop_words,userDictPath).splitString()
    # x_matrix= LDAHelpers(x_data, a).getTFMat()
    # model = lda.LDA(1, n_iter = 1500, random_state=1)
    # model.fit(np.array(x_matrix))
    # skc.printTopWords(model, a, 20,x_label=x_data)
    # # ------------------------------------------------------------------------
    # # 统一分类
    # # skc.getDocTopic(skc.data,skc,8)
    # # 分类打标
    # # sourceData = json.load(open(featuresPath))
    # # skc.getDataSamples(featuresPath)
    # # for sigle in list:
    # #     print '---------->'
    # #     x_lable = [skc.sourceData[i] for i in sigle[0]]
    # #     print x_lable
    # print '-------------------------------------'





























