import json
import h5py
import string
import codecs
import sys
import time

'''
统计文件字符长度 平均值为94.038
'''
def avg_char(file):
    count = 0
    sum = 0
    for line in open(file, 'r', encoding='utf-8'):
        c = line.split(" ")
        sum += len(c)
        count = count + 1
    print(sum/count)

file = "./original_split_char.utf8"
avg_char(file)