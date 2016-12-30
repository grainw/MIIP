#coding=utf-8

from FileUtils import *
from DataFactoryImpl import *
from LDAHelpers import *
from LoggerFacotory import *
import jieba.posseg as pseg
import jieba
import lda
import argparse
#enable parallel for jieba
jieba.enable_parallel()


parser = argparse.ArgumentParser()
parser.add_argument('-f', "--sourcefile", help="number of threads for running", type=str)
parser.add_argument('-s', '--stopwords', help='stop words list', type=str)
parser.add_argument('-u', '--userdict', help='specified user dict', type=str)
args = parser.parse_args()

data = FileUtils(args.sourcefile, FileType.JSON).doRead()
stop_words = FileUtils(args.stopwords, FileType.TEXT, ["words"]).doRead()

#print data.iloc[0]['content']
dict_word = {}
theta = [0.3, 0.5, 0.7]
all_words = []
for i in range(len(data.index)):
    doc = DataFactoryImpl(data.iloc[i:i+1], args.userdict, stop_words).getAllWords()
    all_words.append(doc)
    doc_word = list(set(doc['content'].tolist()[0]))
    X = LDAHelpers(doc, doc_word).getTFMat()
    model = lda.LDA(1, n_iter=1000, random_state=1)
    model.fit(np.array(X))
    for j in theta:
        words = []
        w = 0.0
        for k in model.topic_word_[0].argsort()[:-len(model.topic_word_[0])- 1:-1]:
            if w <= j:
                w += model.topic_word_[0][k]
                words.append(doc_word[k])
            else:
                break
        item = (doc['_id'][0], words, doc['from'][0], doc['url'][0], doc['zhuti'][0])
        if j in dict_word.keys():
            dict_word[j].append(item)
        else :
            dict_word[j] = []
            dict_word[j].append(item)

for key in dict_word:
    pd.DataFrame(dict_word[key], columns=['_id', 'content', 'from', 'url', 'zhuti']).to_csv('testWords' + str(key*100) + '.csv', encoding='utf-8')
pd.concat([i for i in all_words]).to_csv('testAllWords.csv', encoding='utf-8')
