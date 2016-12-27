# coding: utf-8
from functions import *

from ProcessThread import *
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
log.info("generate data")
stop_words_list = loadStopWords('stopwords1.txt')
n_features = 2000
#n_topics = 14440
dxy, label_dxy = getDataSamples('dxy.json')
net, label_39net = getDataSamples('39net2.cvs')
data_samples = dxy + net
data_label = label_dxy+label_39net
#tf with LDA
# def generateWords(data_sample, n_features, stop_words_list, theta):
#     words = []
#     for data in data_sample:
#         allWords = list(set(getAllWords([data], stop_words_list)))
#         X = getWordsTFMat([data], allWords, stop_words_list)
#         model = lda.LDA(1, n_iter=1000, random_state=1)
#         model.fit(np.array(X))
#         w = 0
#         for i in model.topic_word_[0].argsort():
#             if w < theta:
#                 w += model.topic_word_[0][i]
#                 words.append(allWords[i])
#     return words, allWords

# allWords = []
# words, allWords = generateWords(data_samples, n_features, stop_words_list, theta)
step = 500
theta = [0.5,0.6,0.7]
k = len(data_samples)%step
num = 0
if k==0:
    num = len(data_samples)/step
else :
    num = len(data_samples)/step+1
for j in range(num):
        start = j*step
        end = (j+1)*step
        if j==num-1:
            # start = j*step
            end = len(data_samples)
        log.info( '数据分割为:data_sample['+str(start)+':'+str(end)+']')
        thread = LdaThread(j,"ThreadName"+str(j),data_samples[start:end],data_label[start:end], stop_words_list, theta)
        thread.start()
        time.sleep(600)





    # file = open('wordset'+str(theta), 'w')
    # for word in words:
    #     print >> file, word

# file = open('allWords', 'w')
# for word in allWords:
#     print >> file, word
