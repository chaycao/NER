# -*- coding: utf-8 -*-

'''
python pretrain.py input_file cws_info_filePath cws_data_filePath
'''

#2016年 03月 03日 星期四 11:01:05 CST by Demobin

import json
import h5py
import string
import codecs
import sys
import time

# PB:开始；PE:结束；PM：中间；PS:单独; O:其他; X;填充
corpus_tags = [
        'PB','PM','PE','PS','O','X'
    ]

retain_unknown = 'retain-unknown'
retain_empty = 'retain-empty' # 用于填充

def saveTrainingInfo(path, trainingInfo):
    '''保存分词训练数据字典和概率'''
    print('save training info to %s'%path)
    fd = open(path, 'w')
    (vocab, indexVocab) = trainingInfo
    for char in vocab:
        fd.write(str(char.encode('utf-8')) + '\t' + str(vocab[char]) + '\n')
    fd.close()

def loadTrainingInfo(path):
    '''载入分词训练数据字典和概率'''
    print('load training info from %s'%path)
    fd = open(path, 'r')
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = [0 for i in range(len(lines))]
    for line in lines:
        rst = line.strip().split('\t')
        if len(rst) < 2: continue
        char, index = rst[0], int(rst[1])
        vocab[char] = index
        indexVocab[index] = char
    return (vocab, indexVocab)

def saveTrainingData(path, trainingData):
    '''保存分词训练输入样本'''
    print('save training data to %s'%path)
    #采用hdf5保存大矩阵效率最高
    fd = h5py.File(path,'w')
    (X, y) = trainingData
    fd.create_dataset('X', data = X)
    fd.create_dataset('y', data = y)
    fd.close()

def loadTrainingData(path):
    '''载入分词训练输入样本'''
    print('load training data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    y = fd['y'][:]
    fd.close()
    return (X, y)

def sent2vec2(sent, vocab, num):
    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])  # 字在vocabIndex中的索引
        else:
            charVec.append(vocab[retain_unknown])
    # 填充到指定长度
    while len(charVec) < num:
        charVec.append(vocab[retain_empty])
    return charVec

# def sent2vec(sent, vocab, ctxWindows = 5):
#     chars = []
#     for char in sent:
#         chars.append(char)
#     return sent2vec2(chars, vocab, ctxWindows = ctxWindows)

def doc2vec(fname, vocab):
    '''文档转向量'''

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    #样本集
    X = []
    y = []

    #遍历行
    for line in lines:
        #按空格分割
        words = line.strip().split()
        #每行的分词信息
        chars = [] # 存一个个的词
        tags = []  # 词对应的tag
        for word in words:
            rst = word.split('/')
            if len(rst) < 2:
                continue
            word, tag = rst[0], rst[1]

            chars.append(word)
            tags.append(corpus_tags.index(tag))

        #将句子转成词向量，长度短的，填充到指定长度
        lineVecX = sent2vec2(chars, vocab, 128)

        #lab填充到指定长度
        lineVecY = tags
        while len(lineVecY) < 128:
            lineVecY.append(corpus_tags.index('X')) # 填充符用X表示

        # 理论上说，X Y应该都是128维的
        X.extend(lineVecX)
        y.extend(lineVecY)

    return X, y

def vocabAddChar(vocab, indexVocab, index, char):
    if char not in vocab:
        vocab[char] = index
        indexVocab.append(char)
        index += 1
    return index

def genVocab(fname, delimiters = [' ', '\n']):

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    vocab = {}      # 词: 在indexVocab中的索引
    indexVocab = [] # 词

    #遍历所有行
    index = 0
    for line in lines:
        words = line.strip().split()
        if len(words) <= 0: continue
        #遍历所有词
        #如果为分隔符则无需加入字典
        for word in words:
            word = word.strip('[ ')
            end_index = word.find(']')
            if end_index >= 0:
                word = word[0:end_index]
            rst = word.split('/')
            if len(rst) < 2:
                continue
            word, tag = rst[0], rst[1]

            if word not in delimiters:
                index = vocabAddChar(vocab, indexVocab, index, word)

    #加入未登陆新词和填充词
    vocab[retain_unknown] = len(vocab)
    vocab[retain_empty] = len(vocab)
    indexVocab.append(retain_unknown)
    indexVocab.append(retain_empty)
    #返回字典与索引
    return vocab, indexVocab

def load(fname):
    print ('train from file', fname)
    delims = [' ', '\n']
    vocab, indexVocab = genVocab(fname)
    X, y = doc2vec(fname, vocab)
    print (len(X), len(y))
    return (X, y), (vocab, indexVocab)


# main
start_time = time.time()

input_file = "./original_with_tag_people_cut128.utf8"
training_info_filePath = "./ner_training_char.info"
training_data_filePath = "./ner_training_char.data"

(X, y), (vocab, indexVocab) = load(input_file)
# TrainInfo：词向量和词典的相关情况
saveTrainingInfo(training_info_filePath, (vocab, indexVocab))
# TrainData：将字表示为向量和标记
saveTrainingData(training_data_filePath, (X, y))

end_time = time.time()
print("used time : %d s" % (end_time - start_time))