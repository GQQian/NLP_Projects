from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram
from li_ngram import li_ngram
import sys
import operator
import csv
import numpy as linspace
import math

def spell_checker_gt_nrgam(method = 'perplexity'):
    gt_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/spell_checking_task/{}/train.txt".format(topic)
        #test_f = indir_pre + "data/spell_checking_task/{}/test.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test('spell_checking_task')
        train_text[topic] = open(train_f, 'r').read()
        #test_text[topic] = open(test_f, 'r').read()
        gt_ngrams[topic] = gt_ngram(train_text[topic])
        if method == 'perplexity':
            for i in xrange(1,5):
            train_perplexity[topic]=gt_ngram.generate_perplexity(i,train_text[topic])
        print train_perplexity[topic]
