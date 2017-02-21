#coding=utf-8

from SkmeansCluster import *
from numpy import linalg as la
import pickle
#口腔测试

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

skc = SkmeansCluster()
cluster_list = skc.hierarchicalCluster(skc.X,skc.fromzhuti)
dict = {0:'腭',1:'牙',2:'颞颌',3:'腺炎',4:'囊肿',5:'舌',6:'唇',7:'溃疡'}
stop_words = FileUtils(stopWordsPath, FileType.TEXT, ["stopwords"]).doRead()
fromzhuti = FileUtils(fromFilePath, FileType.CSV).doRead()['zhuti']
n_top_words = 15
n_top_likely_topic = 5


def getTestTopWords(distr, words_list, n_top_words):
    new_distr = []
    words = []
    for topic in distr:
        idx = topic.argsort()[:-n_top_words -1:-1]
        new_distr.append(topic[idx])
        words.extend(np.array(words_list)[idx])
    return new_distr, words


def getTrainTopWordsProb(topic_word_distr, test_top_words, wordset):
    topics = []
    for topic in topic_word_distr:
        num = len(test_top_words)
        tmp = []
        for i in range(num):
            if test_top_words[i] in wordset:
                tmp.append(topic[wordset.index(test_top_words[i])])
            else:
                tmp.append(0)
        topics.append(tmp)
    return topics

def getMostLikelyTopic(train, test,n_top_likely_topic):
    dist = []
    for tr in train:
        dist.append(la.norm(np.array(tr)-np.array(test)))
        # dist.append(la.norm(np.cos(np.array(tr),np.array(test))))
    return np.array(dist).argsort()[0:n_top_likely_topic]

def getAllWords(data):
    jieba.load_userdict(userDictPath)
    seg_list = pseg.cut(data)
    words = [normalize(word) for word, flag in seg_list if normalize(word) != '' and normalize(word) not in stop_words and flag in ['n','ns','nsf','nt','nz','nl','ng'] ]
    return words

def getTF(data,all_words,labelF=None):
    mat = []
    arr = []
    fs = []
    # if labelF !=None:
    #     for i in labelF:
    #         fs.extend(i)
    for word in all_words:
        # if labelF !=None:
        #     if word in fs:
        #         arr.append(5)
        #     else:
        #         arr.append(data.count(word))
        # else:
            arr.append(data.count(word))
    mat.append(arr)
    return mat
def getFetures(feturesData,label):
    labelF = []
    sourceData = json.load(open(featuresPath))
    for feture in feturesData:
        edata = str(sourceData[feture]['content'])
        pos = edata.rfind("####")
        rdata = edata[:pos]
        print '分割的数据为'+rdata
        if len(rdata)>0:
            #jieba分词
            # words = getAllWords(rdata)
            words = rdata.split("####")
            if len(words)>0:
                labelF.append([w for w in words if len(w)>0])
            else:
                labelF.append([unicode(dict[int(label)], "utf-8") ])
            # print ' '.join(words)
        else:
            print dict[int(label)]
            labelF.append(dict[int(label)])

    #保存模型
    sp = saveModelPath+label+'fs'
    # os.mkdir(sp)
    saveModel = open( sp, 'w')
    pickle.dump(labelF, saveModel)
    return  labelF

def getNWords(labelF,allwords):
    arr = []
    for segment in labelF:
        for i in segment:
            # print 'i的值为：'+''.join(i)
            allwords.append(i.encode("utf-8"))
            arr.append(i.encode("utf-8"))
    return list(set(allwords)),list(set(arr))


def normalizationMatrix(x,Max,Min):
    if Max == Min:
        return 1
    else:
        return (x - Min) / (Max - Min);

def DescribingSymptoms(label):
    raw_inputS = ''
    new_matrix = []
    x_data =skc.data.ix[cluster_list[int(label)][0]]
    featureList = cluster_list[int(label)][0]
    # labelF = getFetures(featureList,label)
    labelF = pickle.load(open(saveModelPath+label+'fs'))
    allwords = DataFactoryImpl(x_data, skc.stop_words,userDictPath).splitString()
    item = []
    for idx, s in x_data.iterrows():
        words = s['content'].strip().split(' ')
        item.append((s['_id'], words, s['from'], s['url'], s['zhuti']))
    x_data = pd.DataFrame(item, columns=['_id','content', 'from', 'url', 'zhuti'])
    x_data['fetures'] = labelF

    x_matrix= LDAHelpers(x_data, allwords).getTFMat()
    # 将新的数据整理
    nWords,slabel = getNWords(labelF,allwords)
    # feautres = [j[0] for j in labelF]
    # print  '取相关特殊：'+'  '.join(feautres)
    # for ss in slabel:
    #     print ss+' '+str(6)
    print ' '.join(list(set(slabel)))
    while True:
        raw_inputS = raw_input('请输入您详细症状：').replace("\t","").replace(" ","").strip()
        # new_words = getAllWords(raw_inputS)
        # print ','.join(new_words)
        # if len(new_words)>0:
        #     setWord = set(new_words)
        #     sWordSet = list(setWord)
        #     new_matrix= getTF(new_words,sWordSet,labelF)
        #     break
        jieba.analyse.set_idf_path(idfPath)
        jieba.analyse.set_stop_words(stopWordsPath)
        tags = jieba.analyse.extract_tags(raw_inputS,topK=n_top_words,withWeight=True)#allowPOS=('n','ns','nsf','nt','nz','nl','ng')
        dict = {}
        keyWord = []
        if len(tags)>0:
            sorted(tags, key=lambda t: t[1])
            min = tags[len(tags)-1][1]
            max = tags[0][1]
            dict = {}
            for tag in tags:
                dict[tag[0]] = normalizationMatrix(tag[1],max,min)
                keyWord.append(tag[0])
            flag = False
            for kw in keyWord:
                if kw in nWords:
                    flag = True
                    break
            if flag:
                break

    # jieba 代替lda生成向量矩阵
    # new_model = lda.LDA(1, n_iter = 1500, random_state=1)
    # new_model.fit(np.array(new_matrix))
    #转化为新的矩阵
    # avgValue = (np.array(x_matrix).argmax()- np.array(x_matrix).argmin())/2
    # n_matrix= LDAHelpers(x_data, nWords,slabel,avgValue).getTFMat()
    # train_model = lda.LDA(len(cluster_list[int(label)][0]), n_iter = 1500, random_state=1)
    # train_model.fit(np.array(n_matrix))
    train_model = pickle.load(open(saveModelPath+label+'model'))
    #savamodel
    # sp1 = saveModelPath+label+'model'
    # saveModel = open( sp1, 'w')
    # pickle.dump(train_model, saveModel)
    skc.printTopWords(train_model, nWords, 20,x_label=x_data)
    # 组装成新的矩阵
    test_top_words = []
    test_topic_prob = []
    for word in keyWord:
        if word in nWords:
            test_topic_prob.append(dict[word])
            test_top_words.append(word)
    # test_topic_prob, test_top_words =getTestTopWords(new_model.topic_word_ ,sWordSet, n_top_words)
    train_topic_prob = getTrainTopWordsProb(train_model.topic_word_, test_top_words, list(nWords))
    topicIdx = getMostLikelyTopic(train_topic_prob, test_topic_prob,n_top_likely_topic)
    print 'most like docs:'+str(topicIdx)

    print '预测结果：'
    ids =[]
    news_ids = []
    for i in topicIdx:
        # 只获取第一个主题
        # print x_data['zhuti'][train_model.doc_topic_[:,i].argmax()]
        ids.append(x_data['zhuti'][train_model.doc_topic_[:,i].argmax()])
    for id in ids:
        if id not in news_ids:
            news_ids.append(id)
    for i in news_ids:
        print i





while True:
    print '欢迎来到口腔问诊平台,小京欢迎您...'
    title = ''
    for i in dict:
        title += str(i)+":"+dict[i]+" "
    raw_inputF = ''
    while True:
        print title
        raw_inputF = raw_input('请选择您对应的选项标号：').replace("\t","").replace(" ","").strip()
        try:
           if isinstance(int(raw_inputF),int):
            break
        except Exception as e:
            print('输入数据格式有问题')

    print raw_inputF
    DescribingSymptoms(str(raw_inputF))
    continue




