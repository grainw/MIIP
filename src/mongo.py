# -*- coding:utf-8 -*-
import pymongo
from pymongo import MongoClient
from CommonUtils import *
from bson import json_util, ObjectId
import json


"""
    操作 Mongo的类
    eg:
    获取mongo数据库,直接采用静态方法调用
    db = Mongo.get_data_db()
    获取表的信息
    db['all_39_info1']



"""

class Mongo(object):  
    __conn = None
    __db = None

    def __init__(self, dbname = None):
        self.__db = self._get_db(dbname)

    #对外接口
    def get_db(self, dbname = None):
        return self.__db 

    #私有方法
    def _get_db(self, dbname = MONGO_REAL_DATABASE):
        if not self.__conn :
            self._get_conn()

        db_name = dbname or MONGO_DATABASE
        db = self.__conn[db_name]
        return db

    def _get_conn(self):
        host = MONGO_HOST or 'localhost'
        port = MONGO_PORT or 27017
        if  MONGO_AUTH:
            client = MongoClient(host, port)
            db = client[MONGO_DATABASE]
            db.authenticate(MONGO_AUTH_USER, MONGO_AUTH_PASSWORD,mechanism='SCRAM-SHA-1')
            self.__conn = client
        else:
            self.__conn = pymongo.Connection(host, port)

    # data_db
    @staticmethod
    def get_data_db(dbname = MONGO_REAL_DATABASE):
        host = MONGO_HOST or 'localhost'
        port = MONGO_PORT or 27017
        db_name = dbname or MONGO_DATABASE
        if MONGO_AUTH:
            client = MongoClient(host, port)
            db = client[MONGO_DATABASE]
            db.authenticate(MONGO_AUTH_USER, MONGO_AUTH_PASSWORD, mechanism='SCRAM-SHA-1')
            return client[db_name]
        else:
            conn = pymongo.Connection(host, port)
            return conn[db_name]

    '''

    colletionName 表名
    lv1name 一级
    lv2name 二级
    limit 条数
    eg：
        items1 = Mongo.getJsonData('all_39_info1',lv1name='头部',lv2name='鼻')
        items1 = Mongo.getJsonData('all_39_info1',lv1name='头部')
        items1 = Mongo.getJsonData('all_39_info1',lv2name='鼻')
    '''
    @staticmethod
    def getJsonData(collectionName = None,lv1name = None,lv2name = None,limit = None):
        db = Mongo.get_data_db()

        if limit==None:
            limit = db[collectionName].count()
        if lv1name!=None and lv2name!=None:
            return json.loads(json_util.dumps(db[collectionName].find({"lv1name":lv1name,"lv2name":lv2name}).limit(limit),encoding='utf-8'),encoding='utf-8')
        elif lv1name!=None:
            return json.loads(json_util.dumps(db[collectionName].find({"lv1name":lv1name}).limit(limit),encoding='utf-8'),encoding='utf-8')
        elif lv2name!=None:
            return json.loads(json_util.dumps(db[collectionName].find({"lv2name":lv2name}).limit(limit),encoding='utf-8'),encoding='utf-8')

if __name__ == '__main__':
    db = Mongo.get_data_db()
    print db['all_39_info1'].count()
    for item in db['all_39_info1'].find({"lv1name":"头部"}).limit(2):
        s = json.loads(json_util.dumps(item,encoding='utf-8'),encoding='utf-8')
        print s['_id']['$oid']
        # print s['content']
    items = json.loads(json_util.dumps(db['all_39_info1'].find({"lv1name":"头部"}).limit(2),encoding='utf-8'),encoding='utf-8')
    for item in items:
            print item['_id']['$oid']
    print '--------------------'
    items1 = Mongo.getJsonData('all_39_info1',lv1name='头部',lv2name='鼻')
    for item in items1:
            print item['_id']['$oid']

