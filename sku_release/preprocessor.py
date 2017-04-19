# coding=utf-8
import re
import jieba
import numpy as np
from nltk import bigrams
from snownlp import SnowNLP
from collections import Counter


class preprocessor:
    def __init__(self, contents):
        self.contents = contents
        self.corpus1 = []
        self.corpus2 = []
        self.corpus_bi = []
        self.normalized_contents = []

    def replace_word(self, word, str_list):
        for item in str_list:
            word = word.replace(item[0], item[1])
        return word

    def normalize(self, text):
        if isinstance(text, str):
            text = text.decode('utf-8')
        # text = text.decode('utf-8').lower()
        text = text.lower()
        text = SnowNLP(text).han  # 繁体转简体
        text = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5 ]').sub(' ', text)  # 去除标点
        text = re.compile('[0-9]+').sub(' NUM', text)
        text = re.compile('[a-z]+').sub('', text)
        return text

    def tokenize(self, text):
        seg_list = jieba.cut(text, cut_all=False,HMM=True)
        words = '/'.join(seg_list).split('/')
        words = [word for word in words if word.strip() != '']
        return words

    def filter_stopwords(self, text_after_tokenize):
        # vocab = self.filter_vocab(min_df)
        words = filter(lambda x: x not in self.stopwords, text_after_tokenize)
        return words

    @staticmethod
    def get_df_vect(word,corpus2):
        df_vect = np.zeros((len(corpus2), ))
        for i,doc in enumerate(corpus2):
            if word in doc:
                df_vect[i] = 1
            else:
                df_vect[i] = 0

        return df_vect

    @staticmethod
    def count_df(corpus2):
        word_frequencies = [Counter(document) for document in corpus2]
        document_frequencies = Counter()
        map(document_frequencies.update, (word_frequency.keys() for word_frequency in word_frequencies))
        df = sorted(document_frequencies.items(), key=lambda itm: itm[1], reverse=True)
        return df

    @staticmethod
    def calc_ent(x):
        len = x.shape[0]
        x_value_list = set([x[i] for i in range(len)])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / len
            logp = np.log2(p)
            ent -= p * logp

        return ent

    @staticmethod
    def calc_condition_ent(x, y):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            sub_y = y[x == x_value]
            temp_ent = preprocessor.calc_ent(sub_y)
            ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

        return ent

    @staticmethod
    def calc_IG(x,y):
        base_ent = preprocessor.calc_ent(y)
        condition_ent = preprocessor.calc_condition_ent(x, y)
        ent_grap = base_ent - condition_ent

        return ent_grap

    def build_corpus2(self):
        self.normalized_contents = []
        for doc in self.contents:
            norm_content = self.normalize(doc)
            self.normalized_contents.append(norm_content)
            words = self.tokenize(norm_content)
            bigram = list(bigrams(words))
            bigram = [w[0]+'_'+w[1] for w in bigram if w[0] not in self.stopwords and w[1] not in self.stopwords]
            words = self.filter_stopwords(words)
            bigram = words + bigram
            self.corpus2.append(words)
            self.corpus_bi.append(bigram)
        return self.corpus2, self.corpus_bi

    def build_corpus1(self):
        self.corpus1 = [' '.join(doc) for doc in self.corpus2]
        return self.corpus1

    def build_corpus(self, userdict, stopwords):
        if len(userdict)>0:
            jieba.load_userdict(userdict)  # 读取用户自定义词典

        self.stopwords = []
        if len(stopwords) > 0:
            for line in open(stopwords):
                self.stopwords.append(line.strip().decode('utf-8'))

        self.corpus2, self.corpus_bi = self.build_corpus2()
        self.corpus1 = self.build_corpus1()


    # def build_session(self, userdict, stopwords, keyword):
    #     jieba.load_userdict(userdict)  # 读取用户自定义词典
    #     self.stopwords = []
    #     if len(stopwords) > 0:
    #         for line in open(stopwords):
    #             self.stopwords.append(line.strip().decode('utf-8'))
    #
    #     self.stopwords.append(keyword)
    #     self.corpus2 = self.build_corpus2()
    #     self.corpus1 = self.build_corpus1()
    #
    #     for i in range(len(self.contents)):
    #         self.session_list.append(session(self.contents[i],i,self.normalized_contents[i],
    #                                          self.corpus1[i],self.corpus2[i]))
    #     return self.session_list

    # def get_keyword_index(self,keyword):
    #     if len(self.corpus1)==0:
    #         return []
    #     return [session.index for session in self.session_list if keyword in session.content]
    #
    # def get_corpus1_by_index(self, idx):
    #     return [session.corpus for session in self.session_list if session.index in idx]
    #
    # def get_corpus2_by_index(self, idx):
    #     return [session.corpus_list for session in self.session_list if session.index in idx]
    #
    # def get_corpus1_by_keyword(self, keyword):
    #     return [session.corpus for session in self.session_list if keyword in session.content]
    #
    # def get_corpus2_by_keyword(self, keyword):
    #     return [session.corpus_list for session in self.session_list if keyword in session.content]

    def get_contents_by_index(self,idx):
        arr = np.squeeze(np.array(self.contents)[idx])
        return arr.tolist()

    # def get_contents_by_keyword(self,keyword):
    #     return [session.content for session in self.session_list if keyword in session.content]