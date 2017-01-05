#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *
import os





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath ='../data/testmodel/dxyallwith50'
stopWordsPath ='../data/testmodel/stopwords1.txt'
userDictPath ='../data/testmodel/user1.dict'
allWordsPath ='../data/dxy/dxy50.csv'
sourceFilePath ='../data/testmodel/dxy1.json'