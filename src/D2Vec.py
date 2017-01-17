#coding:utf-8
#使用doc2vec 判断文档相似性
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import pandas as pd
import jieba.analyse
import jieba
import sys
import os
sys.path.append(os.path)

#读取文件
data = pd.read_csv('../data/39net/39kqAllallwords.csv')
doc=[]            #输出时使用，用来存储未经过TaggedDocument处理的数据，如果输出document，前面会有u
#构建语料库
corpora_documents = []
for idx, series in data.iterrows():
    words_list = series['content'].split(' ')
    document = TaggedDocument(words=words_list, tags=[idx])
    corpora_documents.append(document)
    doc.append(words_list)
#创建model
model = Doc2Vec(size=50, min_count=1, iter=10)
model.build_vocab(corpora_documents)
model.train(corpora_documents)
print('#########', model.vector_size)
#训练
test_data_1 = ' 两者是同一个东西，白念珠菌是教科书用词，白色念珠菌是医院常用词。多谢提醒，那我们就改成书面正规用语吧。（微生物学教科书上，更是用了白假丝酵母菌这个词，大家注意生活里常说的霉菌感染，一般也是指这个东西）'
test_cut_raw_1 =[]
item2=(pseg.cut(test_data_1))
# item2 = jieba.analyse.extract_tags(test_data_1,topK=10,withWeight=True)
for k in list(item2):
	test_cut_raw_1.append(k.word)
print ' '.join(test_cut_raw_1)
inferred_vector = model.infer_vector(test_cut_raw_1)
sims = model.docvecs.most_similar([inferred_vector], topn=3)
print(sims)  #sims是一个tuples,(index_of_document, similarity)
for i in sims:
	similar=""
	print('################################')
	print i[0]
	print " ".join(doc[i[0]])