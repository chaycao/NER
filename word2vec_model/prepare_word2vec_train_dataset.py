# -*- coding: utf-8 -*-

'''
python prepare_word2vec_train_dataset.py input_file output_file
'''

import os
import sys
import logging
import multiprocessing
import time
import json

start_time = time.time()

# program = os.path.basename(sys.argv[0])
# logger = logging.getLogger(program)
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logger.info("running %s" % ' '.join(sys.argv))

input_file = "./data/小数据5000/original_with_tag.utf8"
output_file =  "./data/小数据5000/original_split.utf8"

output_file_handler = open(output_file, 'w', encoding='utf-8')
for line in open(input_file, 'r', encoding='utf-8'):
    new_line = ''
    words = line.strip().split()
    for word in words:
        word = word.strip('[ ')
        end_index = word.find(']')
        if end_index >= 0:
            word = word[0:end_index]
        word, tag = word.split('/')
        new_line = new_line + word + ' '
    output_file_handler.write(new_line.strip() + '\n')
    output_file_handler.flush()
output_file_handler.close()

end_time = time.time()
print("used time : %d s" % (end_time - start_time))