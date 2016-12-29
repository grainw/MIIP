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
# net, label_39net = getDataSamples('39net2.cvs')
data_samples = dxy #+ net
data_label = label_dxy#+label_39net

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
        time.sleep(60)

