# -*- coding: utf-8 -*-

'''
python train_keras_model.py training_info_filePath training_data_filePath output_keras_model_file output_keras_model_weights_file word2vec_model_file
'''

import numpy as np
import json
import h5py
import codecs
import time
import sys

from pretreat import corpus_tags,loadTrainingInfo,loadTrainingData
import viterbi

from sklearn import model_selection

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping
from keras.layers import crf,ChainCRF

from gensim.models import Word2Vec

def train(trainingInfo, trainingData, modelPath, weightPath, word2vec_model_file):

    (vocab, indexVocab) = trainingInfo
    (X, y) = trainingData

    train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y , train_size=0.9, random_state=1)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    outputDims = len(corpus_tags)
    Y_train = np_utils.to_categorical(train_y, outputDims)
    Y_test = np_utils.to_categorical(test_y, outputDims)
    batchSize = 128
    vocabSize = len(vocab) + 1
    wordDims = 100
    maxlen = 128
    hiddenDims = 100

    w2vModel = Word2Vec.load(word2vec_model_file)
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
    for word, index in vocab.items():
        if word in w2vModel:
            e = w2vModel[word]
        else:
            e = embeddingUnknown
        embeddingWeights[index, :] = e

    print("x_shape:"+str(train_X.shape))
    print("y_shape:" + str(Y_train.shape))
    #LSTM
    model = Sequential()
    model.add(Embedding(output_dim = embeddingDim, input_dim = vocabSize + 1,
    input_length = maxlen, mask_zero = True, weights = [embeddingWeights]))
    model.add(Bidirectional(LSTM(output_dim=hiddenDims, return_sequences=True), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(outputDims, activation="softmax")))

    print(model.summary())

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_acc", patience=3)

    result = model.fit(train_X.reshape(4500,128), Y_train.reshape(-1,maxlen,6), batch_size = batchSize,
                    epochs = 100,
                    validation_data = (test_X.reshape(500,128),Y_test.reshape(-1,maxlen,6)),
                    callbacks=[early_stopping])

    j = model.to_json()
    fd = open(modelPath, 'w')
    fd.write(j)
    fd.close()

    model.save_weights(weightPath)

    return model

# main
training_info_filePath = "./ner_training_char.info"
training_data_filePath = "./ner_training_char.data"
output_keras_model_file = "./ner_keras_model"
output_keras_model_weights_file = "./keras_model_weights"
word2vec_model_file = "./word2vec.model"

print ('Loading vocab...')
start_time = time.time()
trainingInfo = loadTrainingInfo(training_info_filePath)
trainingData = loadTrainingData(training_data_filePath)
print("Loading used time : ", time.time() - start_time)
print ('Done!')

print ('Training model...')
start_time = time.time()
model = train(trainingInfo, trainingData, output_keras_model_file, output_keras_model_weights_file, word2vec_model_file)
print("Training used time : ", time.time() - start_time)
print ('Done!')