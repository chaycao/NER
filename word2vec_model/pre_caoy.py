import os
import sys
import logging
import multiprocessing
import time
import json
import random

def splitChar(input_file, output_file):
    output_file_handler = open(output_file, 'w', encoding='utf-8')
    for line in open(input_file, 'r', encoding='utf-8'):
        new_line = ''
        words = line.strip().split()
        for word in words:
            for char in word:
                new_line = new_line + char + ' '
        output_file_handler.write(new_line.strip() + '\n')
        output_file_handler.flush()
    output_file_handler.close()

def randomSelect(input_file, output_file, num):
    output_file_handler = open(output_file, 'w', encoding='utf-8')
    lines = []
    for line in open(input_file, 'r', encoding='utf-8'):
        lines.append(line)
    slice = random.sample(lines, num)
    for line in slice:
        output_file_handler.write(line.strip() + '\n')
        output_file_handler.flush()
    output_file_handler.close()

'''
把标注变为BMESO
PB、PM、PE、PS、O
'''
def peopleName(input_file, output_file):
    output_file_handler = open(output_file, 'w', encoding='utf-8')
    for line in open(input_file, 'r', encoding='utf-8'):
        new_line = ''
        words = line.strip().split()
        length = len(words)
        temp = 0 # 1：前一个是nr
        for i in range(length):
            rst = words[i].split('/')
            if len(rst) < 2:
                continue
            word, tag = rst[0], rst[1]
            # 人名nr
            if (tag == 'nr'):
                if (temp == 0): #上一个不是人名
                    if ((i + 1<length) and len(words[i + 1].split('/')) == 2 and (words[i + 1].split('/')[1] == 'nr')): #下一个词是人名
                        new_line += word + '/PB '
                    else:
                        new_line += word + '/PS '
                else: # 上一个是人名
                    if ((i + 1 < length) and len(words[i + 1].split('/')) == 2 and (words[i + 1].split('/')[1] == 'nr')):  # 下一个词是人名
                        new_line += word + '/PM '
                    else:
                        new_line += word + '/PE '
                temp = 1
            else:
                temp = 0
                new_line += word + '/O '
        output_file_handler.write(new_line.strip() + '\n')
        output_file_handler.flush()

# 求句子最多的词数
def maxWordNum(filename):
    max = 0
    for line in open(filename, 'r', encoding='utf-8'):
        words = line.split(' ')
        if (len(words) > max):
            max = len(words)
    return max

# 求句子最多的词数
def avgWordNum(filename):
    count = 0
    sum = 0
    for line in open(filename, 'r', encoding='utf-8'):
        words = line.split(' ')
        count += 1
        sum += len(words)
    return sum/count

def overNum(filname, num):
    count = 0
    for line in open(filname, 'r', encoding='utf-8'):
        words = line.split(' ')
        if (len(words) > num):
            count += 1
    return count

'''
把句子截断成固定长度
'''
def cutNum(input_file, output_file, num):
    output_file_handler = open(output_file, 'w', encoding='utf-8')
    for line in open(input_file, 'r', encoding='utf-8'):
        new_line = ''
        words = line.split(' ')
        if(len(words) > num): # 需要截断
            words = words[0:num]
            new_line = " ".join(words)
        else:
            new_line = line
        output_file_handler.write(new_line.strip() + '\n')
        output_file_handler.flush()
# start
start_time = time.time()
input_file = "./data/小数据5000/人名识别/original_with_tag_people.utf8"
output_file =  "./data/小数据5000/人名识别/original_with_tag_people_cut128.utf8"

# content
#cutNum(input_file, output_file, 128)
print (maxWordNum(output_file))
# content end

end_time = time.time()
print("used time : %d s" % (end_time - start_time))