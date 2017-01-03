#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *
import os





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath ='../data/testmodel/train_model'
stopWordsPath ='../data/testmodel/stopwords1.txt'
userDictPath ='../data/testmodel/user.dict'
allWordsPath ='../data/testmodel/word_set'
sourceFilePath ='../data/testmodel/dxy1.json'