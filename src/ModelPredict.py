#coding=utf-8

from CommonUtils import *
from FileUtils import *
from DataFactoryImpl import *
from LDAHelpers import *
import numpy as np
from numpy import linalg as la
import lda

'''
load model to predict or accuray
'''
class ModelPredict:
    def __init__(self,sourceFilePath):
        self.sourceFilePath = sourceFilePath
        self.n_top_words = 10
        self.n_top_likely_topic = 5
        self.list_models = None
        self.data_samples = None
        self.data_label = None
        self.stop_words = None
        self.train_all_words = pickle.load(open(allWordsPath))
        self.listdoc_words  = None
        self.zhuti = None
        self.docIdxs = None
        self.features = 15
        self.trainModel  = pickle.load(open(modelPath))
    '''
    input sourceFilePath
    output X matrix tf
    '''
    def processData(self):
        data_samples = FileUtils(self.sourceFilePath, FileType.JSON).doRead()
        self.data_label = data_samples['label']
        self.zhuti = data_samples['zhuti']
        stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
        self.data_samples = data_samples
        self.stop_words = stop_words
        list_models = []
        list_doc_words = []
        for i in range(len(data_samples.index)):
            doc = DataFactoryImpl(data_samples.iloc[i:i+1], stop_words,userDictPath).getAllWords()
            doc_words =  list(set(doc['content'].tolist()[0]))
            list_doc_words.append(doc_words)
            X = LDAHelpers(doc, doc_words).getTFMat()
            model = lda.LDA(1, n_iter=1500, random_state=1)
            model.fit(np.array(X))
            list_models.append(model)
        self.list_models = list_models
        self.listdoc_words = list_doc_words

    def getTestTopWords(self,distr, words_list, n_top_words):
        new_distr = []
        words = []
        for topic in distr:
            idx = topic.argsort()[:-n_top_words -1:-1]
            new_distr.append(topic[idx])
            words.extend(np.array(words_list)[idx])
        return new_distr, words

    def getTrainTopWordsProb(self,topic_word_distr, test_top_words, wordset):
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

    def getMostLikelyTopic(self,train, test,n_top_likely_topic):
        dist = []
        for tr in train:
            dist.append(la.norm(np.array(tr)-np.array(test)))
        return np.array(dist).argsort()[0:-n_top_likely_topic-1:-1]

    def predict(self):
        #load model
        #加入lda
        docIdxs =[]
        for m in range(len(self.list_models)):
            sTopic_Word = self.list_models[m].topic_word_
            sWordSet = self.listdoc_words[m]
            test_topic_prob, test_top_words =self.getTestTopWords(sTopic_Word ,sWordSet, self.n_top_words)
            train_topic_prob = self.getTrainTopWordsProb(self.trainModel.topic_word_, test_top_words, list(self.train_all_words))
            topicIdx = self.getMostLikelyTopic(train_topic_prob, test_topic_prob,self.n_top_likely_topic)
            log.info('most like docs:'+str(topicIdx))
            for idx in topicIdx:
                log.info(" ".join([list(self.train_all_words)[i]
                                for i in self.trainModel.topic_word_[idx].argsort()[:-self.features- 1:-1]]))
            for i in topicIdx:
                log.info(self.zhuti[self.trainModel.doc_topic_[:,i].argmax()])
            for k in range(self.n_top_likely_topic):
                if k ==0:
                    docIdx = self.trainModel.doc_topic_[:,topicIdx[k]].argmax()
                    docIdxs.append(docIdx)
                    break
        self.docIdxs = docIdxs
        return docIdxs

    def accuray(self):
        test_label = np.array(self.data_label).ravel()
        train_label = np.array(self.docIdxs)
        result = (test_label == train_label)   # True则预测正确，False则预测错误
        c = np.count_nonzero(result)    # 统计预测正确的个数
        log.info(c)
        log.info('Accuracy: %.2f%%' % (100 * float(c) / float(len(result))))


if __name__ == '__main__':
    log.info('begin to load model')
    mp = ModelPredict(sourceFilePath)
    mp.processData()
    mp.predict()
    mp.accuray()
    log.info('ending to predict model')







