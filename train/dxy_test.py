# coding: utf-8

from functions import *
from dxy_train import trainModel
import os

print "generate data"
stop_words_list = loadStopWords('stopwords1.txt')
n_features = 2000
n_top_words = 20
n_topics = 24
dxy, label_dxy = getDataSamples('dxy1.json')
# net, label_39net = getDataSamples('39net2.cvs')
# data_samples = dxy + net
# zhuti = label_dxy + label_39net
data_samples = dxy
zhuti = label_dxy
##Save  label to user_dict
# with open("user_dict", "w") as f:
#     for i in zhuti:
#         print >> f, i

#test_case, zhuti = getDataSamples('dxy1.json')
#data_samples = data_samples[0:100]

#Train model
train , train_words = trainModel(data_samples, n_top_words, stop_words_list, n_features, n_topics, save=False)
printTopWords(train, train_words, n_top_words)
#load train data
# print "load training data"
# saved_path = os.getcwd() + '/samples-100-2016-12-14-16:39:22'
# train       = pickle.load(open(saved_path + '/train_model'))
# train_words = pickle.load(open(saved_path + '/word_set'))
#generate test case

# #tf with LDA
# n_topics = 1
# test_case = [test_case[-1]]
# words = list(set(getAllWords(test_case, stop_words_list)))
# X = getWordsTFMat(test_case, words, stop_words_list)
# model = lda.LDA(1, n_iter=1500, random_state=1)
# model.fit(np.array(X))
# printTopWords(model, words, n_top_words)
#
# #tfidf with NMF
# test_topic_prob, test_top_words = getTestTopWords(model.topic_word_, words, 10)
# train_topic_prob = getTrainTopWordsProb(train.topic_word_, test_top_words, list(train_words))
# print test_topic_prob
# topicIdx = getMostLikelyTopic(train_topic_prob, test_topic_prob)
# print "Most likely topic is: ", topicIdx
# for idx in topicIdx:
#     print " ".join([list(train_words)[i]
#                             for i in train.topic_word_[idx].argsort()[:-20- 1:-1]])
#     #print train.topic_word_[idx]
#     print train_topic_prob[idx]
# #printTopWords(train, train_words, 20)
# print "Most related docs:"
# docIdx = train.doc_topic_[:,topicIdx[0]].argmax()
# #print zhuti[docIdx]# "  ", "http://dxy.com" + url[docIdx]
# #docIdx = getMostRelatedDocs(train.doc_topic_, topicIdx, zhuti)
