# -*- coding: utf-8 -*-

from keras.models import model_from_json
import numpy as np
import json
import h5py
import codecs
import time
import sys
from pretreat import corpus_tags,loadTrainingInfo,loadTrainingData
from keras.utils import np_utils
from gensim.models import Word2Vec

modelPath = './ner_keras_model'
weightPath = './keras_model_weights'
data_filePath = './ner_training_char.data'
info_filePath = './ner_training_char.info'
word2vec_model_file = './word2vec.model'

# 加载数据
Info = loadTrainingInfo(info_filePath)
Data = loadTrainingData(data_filePath)
(vocab, indexVocab) = Info
(X, y) = Data

X = np.array(X)
y = np.array(y)
outputDims = len(corpus_tags)

batchSize = 128
vocabSize = len(vocab) + 1
wordDims = 100
maxlen = 128
hiddenDims = 100

# 加载模型
fd = open(modelPath, 'r')
j = fd.read()
fd.close()
model = model_from_json(j)
model.load_weights(weightPath)
X = X.reshape(-1,batchSize)
p_y = model.predict(X, batchSize, verbose=1)

y = y.reshape(-1,batchSize)

n = 0; # 数据中的数量
find_n = 0; # 预测找到的数量
right_n = 0; # 预测对的数量

for i in range(len(p_y)):   #句子
    for j in range(len(p_y[0])):  #词
        lable = np.argmax(p_y[i][j])
        if (lable == 5): #填充符，到句尾，跳下一个句子
            break
        if (y[i][j] <= 3):
            n += 1
            if (lable == y[i][j]):
                right_n += 1
        if (lable <= 3):
            find_n += 1

recall = right_n / n
acc = right_n / find_n
f1 = 2 * recall * acc / (recall + acc)
print ('召回率：' + str(recall))
print ('准确率：' + str(acc))
print ('F1：' + str(f1))
    
        