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
        self.n_top_words = 15
        self.n_top_likely_topic = 10
        self.list_models = None
        self.data_samples = None
        self.data_label = None
        self.stop_words = None
        # self.train_all_words = pickle.load(open(allWordsPath))
        self.train_all_words = None
        self.listdoc_words  = None
        self.zhuti = None
        self.docIdxs = None
        self.predict_zhuti = []
        self.features = 15
        self.trainModel  = pickle.load(open(modelPath))
        self.dictVail = []
        self.num = 2
        self.fromzhuti = None
    '''
    input sourceFilePath
    output X matrix tf
    '''
    def processData(self):
        data_samples = FileUtils(self.sourceFilePath, FileType.JSON).doRead()
        # self.data_label = data_samples['zhuti']

        self.zhuti = data_samples['zhuti']
        self.stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
        # self.train_all_words =FileUtils(allWordsPath, FileType.CSV).doRead()['content']
        self.train_all_words = DataFactoryImpl(FileUtils(allWordsPath, FileType.CSV).doRead(),self.stop_words).splitString()
        self.fromzhuti = FileUtils(fromFilePath, FileType.JSON).doRead()['zhuti']
        self.data_samples = data_samples
        list_models = []
        list_doc_words = []
        for i in range(0,len(data_samples.index)/self.num):
            doc = DataFactoryImpl(data_samples.iloc[i:i+1], self.stop_words,userDictPath).getAllWords()
            log.info("doc"+str(doc))
            if len(doc['content'].tolist()[0])==0:
                self.dictVail.append(i)
                continue
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
        return np.array(dist).argsort()[0:n_top_likely_topic]

    def predict(self):
        #load model
        #加入lda
        docIdxs =[]
        for m in range(len(self.list_models)):
            sTopic_Word = self.list_models[m].topic_word_
            sWordSet = self.listdoc_words[m]
            test_topic_prob, test_top_words =self.getTestTopWords(sTopic_Word ,sWordSet, self.n_top_words)
            for m in test_top_words:
                print m
            train_topic_prob = self.getTrainTopWordsProb(self.trainModel.topic_word_, test_top_words, list(self.train_all_words))
            topicIdx = self.getMostLikelyTopic(train_topic_prob, test_topic_prob,self.n_top_likely_topic)
            log.info('most like docs:'+str(topicIdx))
            # for idx in topicIdx:
            #     log.info(" ".join([list(self.train_all_words)[i]
            #                     for i in self.trainModel.topic_word_[idx].argsort()[:-self.features- 1:-1]]))
            for i in topicIdx:
                # 只获取第一个主题
                log.info(self.fromzhuti[self.trainModel.doc_topic_[:,i].argmax()])
            for i in topicIdx:
                # 只获取第一个主题
                # log.info(self.fromzhuti[self.trainModel.doc_topic_[:,i].argmax()])
                self.predict_zhuti.append(self.fromzhuti[self.trainModel.doc_topic_[:,i].argmax()])
                break
        #     for k in range(self.n_top_likely_topic):
        #         if k ==0:
        #             docIdx = self.trainModel.doc_topic_[:,topicIdx[k]].argmax()
        #             docIdxs.append(docIdx)
        #             break
        # self.docIdxs = docIdxs
        # return docIdxs

    def accuray(self):
        # test_label = np.array(self.data_label).ravel()
        test_predict  = []
        log.info("总数量为"+str(len(self.zhuti[0:len(self.data_samples.index)/self.num])))
        log.info("过滤后总数为"+str(len(self.dictVail)))
        log.info("过滤的序号为："+str(self.dictVail))
        if len(self.dictVail)>0:
            k = 0
            for z in self.zhuti[0:len(self.data_samples.index)/self.num]:
                if k not in self.dictVail:
                    test_predict.append(z)
                k = k+1
            test_label =  np.array(test_predict).ravel()
        else:
            test_label = np.array( self.zhuti).ravel()
        ##应用主题打标
        # train_label = np.array(self.docIdxs)
        train_label = np.array(self.predict_zhuti)
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







