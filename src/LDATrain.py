#coding=utf-8
from FileUtils import *
from DataFactoryImpl import *
from LDAHelpers import *
from LoggerFacotory import *
import jieba.posseg as pseg
import jieba
import lda
import pickle
import argparse
#enable parallel for jieba
jieba.enable_parallel()

def printTopWords(model, allWords, n_top_words):
    for topicIdx, topic in enumerate(model.topic_word_):
        print("Topic #%d:" % topicIdx)
        print(" ".join([allWords[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        docIdx = model.doc_topic_[:, topicIdx].argmax()
        print "Doc: ", data['zhuti'][docIdx], ' ', data['_id'][docIdx]

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--data",          help="data file",                   type=str)
parser.add_argument('-s', '--stopwords',     help='stop words list',             type=str)
parser.add_argument('-w', '--weightedwords', help='weighted words for per data', type=str)
parser.add_argument('-f', '--filename',      help='filename for output',         type=str)
args = parser.parse_args()

data          = FileUtils(args.data, FileType.CSV).doRead()
reduced_data  = FileUtils(args.weightedwords, FileType.CSV).doRead()
stop_words    = FileUtils(args.stopwords, FileType.TEXT, ["words"]).doRead()
all_words     = DataFactoryImpl(reduced_data, stop_words).splitString()
X             = LDAHelpers(data, all_words).getTFMat()
model         = lda.LDA(len(data.index), n_iter = 1500, random_state=1)
model.fit(np.array(X))
pickle.dump(model, args.filename)

printTopWords(model, all_words, 20)
