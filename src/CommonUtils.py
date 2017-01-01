#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath = '../train/samples-1-2017-01-01-15-03-55/train_model'
stopWordsPath = '../train/stopwords1.txt'
userDictPath = '../train/user.dict'
allWordsPath = '../train/samples-1-2017-01-01-15-03-55/word_set'
sourceFilePath = '../train/dxy1.json'