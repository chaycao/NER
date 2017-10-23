# -*- coding: utf-8 -*-

'''
python test_keras_model.py training_info_file keras_model_file keras_model_weights_file test_data_file output_file
'''

import numpy as np
import json
import h5py
import codecs
import time
import sys

import pretreat
import viterbi

from sklearn import model_selection

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
# from keras.models import Sequential,Graph, model_from_json
# from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

from gensim.models import Word2Vec

def loadModel(modelPath, weightPath):

    fd = open(modelPath, 'r')
    j = fd.read()
    fd.close()

    model = model_from_json(j)

    model.load_weights(weightPath)

    return model


# 根据输入得到标注推断
def testSent(sent, model, trainingInfo):
    (initProb, tranProb), (vocab, indexVocab) = trainingInfo
    vec = pretreat.sent2vec(sent, vocab, ctxWindows = 7)
    vec = np.array(vec)
    probs = model.predict_proba(vec)
    #classes = model.predict_classes(vec)

    prob, path = viterbi.viterbi(vec, pretreat.corpus_tags, initProb, tranProb, probs.transpose())

    ss = ''
    for i, t in enumerate(path):
        ss += '%s/%s '%(sent[i], pretreat.corpus_tags[t])
    # ss = ''
    # word = ''
    # for i, t in enumerate(path):
    #     if cws.corpus_tags[t] == 'S':
    #         ss += sent[i] + ' '
    #         word = ''
    #     elif cws.corpus_tags[t] == 'B':
    #         word += sent[i]
    #     elif cws.corpus_tags[t] == 'E':
    #         word += sent[i]
    #         ss += word + ' '
    #         word = ''
    #     elif cws.corpus_tags[t] == 'M': 
    #         word += sent[i]

    return ss

def testFile(fname, dstname, model, trainingInfo):
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    fd = open(dstname, 'w')
    for line in lines:
        rst = testSent(line.strip(), model, trainingInfo)
        fd.write(rst.encode('utf-8') + '\n')
    fd.close()

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    training_info_file, keras_model_file, keras_model_weights_file, test_data_file, output_file = sys.argv[1:6]

    training_info = pretreat.loadTrainingInfo(training_info_file)
    print ('Loading model...')
    start_time = time.time()
    model = loadModel(keras_model_file, keras_model_weights_file)
    print("Loading used time : ", time.time() - start_time)
    print ('Done!')
    print ('-------------start predict----------------')
    # s = u'为寂寞的夜空画上一个月亮'
    # print (testSent(s, model, cwsInfo))
    testFile(test_data_file, output_file, model, training_info)
