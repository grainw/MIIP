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
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class SkmeansCluster:
    def __init__(self):
        self.fromzhuti = FileUtils(fromFilePath, FileType.CSV).doRead()['zhuti']
        data = FileUtils(fromFilePath, FileType.CSV).doRead()
        stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
        allWords = DataFactoryImpl(data, stop_words,userDictPath).splitString()
        self.X = LDAHelpers(data, allWords).getTFMat()
        # trainModel  = pickle.load(open(modelPath))
        # X = trainModel.doc_topic_

    def kmeansCluster(self,clusters):
        kmeans = KMeans(n_clusters=clusters,max_iter=1000,tol=1e-7, random_state=0).fit(self.X)
        list = []
        for i in range(clusters):
            list.append(np.where(kmeans.labels_==i))
        self.printClusterIndexToLabel(list)
        return list

    def printClusterIndexToLabel(self,list):
        for j in list:
            arr = []
            for k in j[0]:
                arr.append(self.fromzhuti[k])
                # arr.append(fromzhuti[trainModel.doc_topic_[:,k].argmax()])
            print ",".join(arr)

    def hierarchicalCluster(self,flag=False):
        disMat = sch.distance.pdist(self.X,'cosine')
        Z=sch.linkage(disMat,method='average')
        P=sch.dendrogram(Z)
        plt.savefig('plot_dendrogram.png')
        #根据linkage matrix Z得到聚类结果:
        cluster= sch.fcluster(Z,t=0.74,criterion='distance',depth=6)
        print "Original cluster by hierarchy clustering:\n",cluster
        print max(cluster)
        print '层次聚类的索引结果...................'
        list = []
        for i1 in range(1,(max(cluster)+1)):
            list.append(np.where(cluster==i1))
        self.printClusterIndexToLabel(list)
        if flag:
            save_path = os.getcwd() +'\\cluster-'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            os.mkdir(save_path)
            saveModel = open(save_path + '\\cluster_model', 'w')
            pickle.dump(list, saveModel)
        return list
if __name__ == '__main__':
    skc = SkmeansCluster()
    skc.hierarchicalCluster(True)















