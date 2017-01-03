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
parser.add_argument('-d', "--data",      help="data file",            type=str)
parser.add_argument('-s', '--stopwords', help='stop words list',      type=str)
parser.add_argument('-u', '--userdict',  help='specified user dict',  type=str)
parser.add_argument('-f', '--filename',  help='file name for output', type=str)
args = parser.parse_args()

data = FileUtils(args.data, FileType.JSON).doRead()
stop_words = FileUtils(args.stopwords, FileType.TEXT, ["words"]).doRead()

#print data.iloc[0]['content']
dict_word = {}
theta = [0.3, 0.5, 0.7]
all_words = []
for i in range(len(data.index)):
    doc = DataFactoryImpl(data.iloc[i:i+1], stop_words, args.userdict).getAllWords()
    doc_word = list(set(doc['content'].tolist()[0]))
    X = LDAHelpers(doc, doc_word).getTFMat()
    all_words.append((doc['_id'][0], ' '.join(doc['content']), doc['from'][0], doc['url'][0], doc['zhuti'][0]))
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
        item = (doc['_id'][0], ' '.join(words), doc['from'][0], doc['url'][0], doc['zhuti'][0])
        if j in dict_word.keys():
            dict_word[j].append(item)
        else :
            dict_word[j] = []
            dict_word[j].append(item)


for key in dict_word:
    pd.DataFrame(dict_word[key], columns=['_id', 'content', 'from', 'url', 'zhuti']).to_csv(args.filename + str(int(key*100)) + '.csv', encoding='utf-8')

pd.DataFrame(all_words, columns=['_id', 'content', 'from', 'url', 'zhuti']).to_csv(args.filename + 'allwords.csv', encoding='utf-8')
