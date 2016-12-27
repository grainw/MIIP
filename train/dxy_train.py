# coding: utf-8

from functions import *
import os
import time

def trainModel(data_samples, n_top_words, stop_words_list, n_features, n_topics, save=False):
    n_samples = len(data_samples)
    #n_topics = n_samples
    wordset = preprocessor(data_samples, n_features, stop_words_list)
    X = getWordsTFMat(data_samples, wordset, stop_words_list)
    print("Extracting tf features for LDA...")
    print "number of words: ", len(wordset)

    model = lda.LDA(n_topics, n_iter=1500, random_state=1)
    model.fit(np.array(X))
    printTopWords(model,list(wordset), n_top_words)
    if save:
        save_path =  'samples-' + str(n_samples) + '-' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        os.mkdir(save_path)
        # topic_word = open(save_path + '/train_model', 'wr')
        # doc_topic = open(save_paht + '/train_model', 'wr')
        # np.save(topic_word, np.array(model.topic_word_))
        # np.save(doc_topic, np.array(model.doc_topic_))
        saveModel = open(save_path + '/train_model', 'w')
        pickle.dump(model, saveModel)
        saveWordSet = open(save_path + '/word_set', 'w')
        pickle.dump(wordset, saveWordSet)
    return model, wordset
