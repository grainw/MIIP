#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *
import os





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath ='../data/testmodel/39kqallTrainModel'
stopWordsPath ='../data/common/stopwords'
userDictPath ='../data/testmodel/user1.dict'
allWordsPath ='../data/testmodel/39kqWith50TrainWords'
sourceFilePath ='../data/testmodel/testdxy.json'
fromFilePath = '../data/39net/39kqAllallwords.csv'