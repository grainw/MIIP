#!/usr/bin/python
# -*- coding: UTF-8 -*-

from LoggerFacotory import *





log = Logger('milp.log', logging.INFO, logging.INFO)


modelPath = '../train/model'
stopWordsPath = '../train/stopwords1.txt'
userDictPath = '../train/user.dict'
allWordsPath = '../train/allwords'
sourceFilePath = '../train/dxy.json'