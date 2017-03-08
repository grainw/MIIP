#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *
import os





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath ='../data/testmodel/39kqallTrainModel'
stopWordsPath ='../data/common/stopwords'
userDictPath ='../data/testmodel/user1.dict'
idfPath = '../data/common/idf1.txt'
allWordsPath ='../data/testmodel/39kqWith50TrainWords'
sourceFilePath ='../data/testmodel/testdxy.json'
fromFilePath = '../data/39net/39kqAllallwords.csv'
# fromFilePath = '../data/39net/yanhouallwords.csv'
featuresPath = '../data/39net/39kq.txt'
saveModelPath = '../data/modeldata1/'

MONGO_HOST = '172.24.8.15'
MONGO_PORT = 27017
MONGO_AUTH_USER = 'manager'
MONGO_AUTH_PASSWORD = 'mongodb123'
MONGO_AUTH = True
MONGO_DATABASE = 'admin'
MONGO_REAL_DATABASE = 'crawler'

