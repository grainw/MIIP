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
        self.model = None
        self.data_samples = None
        self.data_label = None
        self.stop_words = None
        self.train_all_words = None
        self.doc_words  = None
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
        stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
        train_all_words = FileUtils(allWordsPath, FileType.CSV, ["allwords"]).doRead()
        self.data_samples = data_samples
        self.stop_words = stop_words
        self.train_all_words = train_all_words
        for i in range(len(data_samples.index)):
            doc = DataFactoryImpl(data_samples.iloc[i:i+1], userDictPath, stop_words).getAllWords()
            doc_words =  list(set(doc['content'].tolist()[0]))
            X = LDAHelpers(doc, doc_words).getTFMat()
            model = lda.LDA(1, n_iter=1500, random_state=1)
            model.fit(np.array(X))
            self.model = model
            self.doc_words = doc_words

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

    def getMostLikelyTopic(train, test,n_top_likely_topic):
        dist = []
        for tr in train:
            dist.append(la.norm(np.array(tr)-np.array(test)))
        return np.array(dist).argsort()[0:-n_top_likely_topic-1:-1]

    def predict(self):
        #load model
        test_topic_prob, test_top_words =self.getTestTopWords(self.model.topic_word_, self.doc_words, self.n_top_words)
        train_topic_prob = self.getTrainTopWordsProb(self.trainModel.topic_word_, test_top_words, list(self.train_all_words))
        topicIdx = self.getMostLikelyTopic(train_topic_prob, test_topic_prob,self.n_top_likely_topic)
        for idx in topicIdx:
                print " ".join([list(self.train_all_words)[i]
                            for i in self.trainModel.topic_word_[idx].argsort()[:-self.features- 1:-1]])
        docIdxs =[]
        for k in range(self.n_top_likely_topic):
            docIdx = self.trainModel.doc_topic_[:,topicIdx[k]].argmax()
            docIdxs.append(docIdx)
        log.info("most like docs:")
        log.info(docIdxs)
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







