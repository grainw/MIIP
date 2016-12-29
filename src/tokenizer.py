#coding=utf-8

from FileUtils import *
from DataFactoryImpl import *
from LDAHelpers import *
from LoggerFacotory import *
import jieba.posseg as pseg
import jieba
#enable parallel for jieba
jieba.enable_parallel()


data = FileUtils('../train/dxy1.json', FileType.JSON).doRead()
stop_words = FileUtils('../train/stopwords1.txt', FileType.TEXT, ["words"]).doRead()
df = DataFactoryImpl(data, '../train/user.dict', stop_words)
all_words = df.getAllWords()
