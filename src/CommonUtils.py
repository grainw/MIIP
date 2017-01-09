#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *
import os





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath ='../data/testmodel/dxyTrainModel'
stopWordsPath ='../data/testmodel/stopwords1.txt'
userDictPath ='../data/testmodel/user1.dict'
allWordsPath ='../data/testmodel/dxyTrainallWords'
sourceFilePath ='../data/testmodel/testdxy.json'
fromFilePath = '../data/dxy/dxy.json'