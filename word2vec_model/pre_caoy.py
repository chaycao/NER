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

# start
start_time = time.time()
input_file = "./dataset/小数据5000/original_split.utf8"
output_file =  "./dataset/小数据5000/original_split_word.utf8"

# content
splitChar(input_file, output_file)
# content end

end_time = time.time()
print("used time : %d s" % (end_time - start_time))