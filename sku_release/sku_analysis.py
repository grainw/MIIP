# coding=utf-8

from file_IO import file_IO
import numpy as np
from preprocessor import preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lo import lo


if __name__ == '__main__':
    ## 读取数据
    contents = file_IO.read_xls('sku.xlsx')
    comments = contents[1].astype('unicode').values #用户评论
    good_mid_bad = contents[3].values #评论等级
    labels = np.zeros((len(comments),), dtype=np.float)
    labels[good_mid_bad == 'good'] = 2
    labels[good_mid_bad == 'mid'] = 1
    labels[good_mid_bad == 'bad'] = 0

    # 对用户评论进行预处理(包括大小写转换，繁体转简体，去除标点，去除数字，分词，去停用词等)
    pre = preprocessor(comments)
    pre.build_corpus('user.dict', 'stopwords.txt')

    # 利用信息增益对分词结果进行精炼，只选出比较重要的词，该部分可以先不管
    corpus2 = pre.corpus2
    df = preprocessor.count_df(corpus2)
    df = filter(lambda itm: itm[1] > 3, df)
    vocab = [itm[0] for itm in df]
    igs = np.zeros((len(vocab),))
    for i, word in enumerate(vocab):
        x = preprocessor.get_df_vect(word, corpus2)
        ig = preprocessor.calc_IG(x, labels)
        igs[i] = ig
    res = zip(vocab, [itm[1] for itm in df], igs)
    res = sorted(res, key=lambda itm: itm[2], reverse=True)
    igs = np.array([itm[2] for itm in res])
    cs = np.cumsum(igs)
    threshold = 1.1
    selected_idx = cs < threshold * cs[-1]

    #去除不重要的词，该部分可以先不管
    controlled_vocab = dict(zip(lo.sub_bool(vocab, selected_idx), range(np.sum(selected_idx))))
    corpus1 = [' '.join(filter(lambda x: x in controlled_vocab.keys(), doc)) for doc in corpus2]
    valid_idx = lo.valid_idx(corpus1)
    corpus1 = lo.sub_int(corpus1, valid_idx)

    # 将分词后的语料进行tfidf向量化
    vectorizer = TfidfVectorizer(lowercase=False, vocabulary=controlled_vocab,
                                 tokenizer=lambda x: x.split(' '), sublinear_tf=True)
    vectorizer_fit = vectorizer.fit_transform(corpus1)
    tfidf = vectorizer_fit.toarray()

    #由于语料中，中评和差评数据很少，为了保证模型精度，需要重复使用中评和差评数据
    valid_labels = labels[valid_idx]
    tfidf_ext = tfidf
    valid_labels_ext = valid_labels
    for i in range(10):
        tfidf_ext = np.concatenate((tfidf_ext, tfidf[valid_labels == 0], tfidf[valid_labels == 1]))
        valid_labels_ext = np.concatenate(
            (valid_labels_ext, valid_labels[valid_labels == 0], valid_labels[valid_labels == 1]))

    #建立逻辑回归模型
    LR = LogisticRegression()
    LR.fit(tfidf_ext, valid_labels_ext)

    #用逻辑回归模型进行预测
    pred = LR.predict(tfidf)

    #保存结果到xls文件中
    valid_comments = lo.sub_int(comments, valid_idx)
    flag = valid_labels == pred
    file_IO.write_xls((valid_comments, valid_labels, pred, corpus1, flag.tolist()),
                      'sku_prd.xls', ['comment', 'label', 'pred_label', 'corpus', 'flag'])