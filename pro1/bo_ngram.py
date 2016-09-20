# backoff ngram
import os
import random
from gt_ngram import gt_ngram
import operator
import preprocess
from math import pow, log, exp
import numpy as np

import li_gram


class bo_ngram(gt_ngram):
    def __init__(self,content,r = [0.05,0.25,0.7]):
        super(bo_ngram,self).__init__(content)
        self.r = np.array(r)

# only one iteration
    def generate_perplexity(self, n, sentences):
        for z in xrange(3):
            x = z + 1
            self.nprob_dic[x] = self.nprob_dic[x] if x in self.nprob_dic else self.generate_ngram(x)
        tokens = preprocess.preprocess_text(sentences)
        # tokens = preprocess.preprocess_text(sentences).split()

        token_list = [[],[],[]]
        token_list[0] = tokens.replace('<s>', '').split()
        token_list[1]  = tokens.split()
        token_list[2] = tokens.replace('<s>', '<s1> <s2>').split()
        tokens = tokens.split()
        # for keys in self.nprob_dic.keys():
        # use unk_1 to repalce word not in ncounter_dic[1]
        self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
        for eachlist in token_list:
            for i, token in enumerate(eachlist):
                key = tuple([token])
                if key not in self.ncounter_dic[1]:
                    eachlist[i] = '<unk_1>'

        # calculate perplexity
        perp = 0
        
        _len = len(token_list[0])
        bi_fix = 0  # count <unk> in bigram
        tri_fix = 0 # count <unk> in trigram

        
        iters = [0, 0, 0]

        while iters[0] < _len:
            prob = 0 
            prob_tup = []
            for j in xrange(n):
                if j < 1:
                    key = tuple(token_list[j][iters[j]: iters[j] + 1])
                    while '</s>' in key:
                        
                        key = tuple(token_list[j][iters[j]: iters[j] + 1])
                        iters[j] += 1
                    # print "key is {}, iterater is {}, j is {}".format(key, iters[j], j)
                    iters[j] += 1

                else:
                    
                    key = tuple(token_list[j][iters[j]: iters[j] + j + 1])

                    while '</s>' in key:
                        
                        key = tuple(token_list[j][iters[j] : iters[j] + j + 1])
                        iters[j] += 1
                    # print "key is {}, iterater is {}, j is {}".format(key, iters[0], j)

                    # Replace unknown word with <unk> #

                    unk = '<unk_{}>'.format(j)
                    if key not in self.nprob_dic[j + 1]:
                        key = tuple([unk, tokens[-1]])
                        # print "key is {}, iterater is {}, j is {}".format(key, iters[0], j)

                    iters[j] += 1
                try:                    
                    prob_element = self.nprob_dic[j+1][key]
                except KeyError:
                    prob_element = 1
                prob_tup.append(prob_element)
                # print "prob_element is {}, iterater is {}, j is {}".format(key, iters[j], j)
            prob_tup = np.array(prob_tup)
            prob = prob_tup.dot(self.r)
            perp -= log(prob)
        perp = exp(1.0 * perp / len(tokens))
        return perp




################ will be abandoned ###############

        # for i in xrange(_len):
        #     prob = 0 
        #     prob_tup = []

        #     for j in xrange(n):
        #         # ntemp: helps get all unigram to n-gram
        #         ntemp = j + 1
        #         if ntemp < 2:
           #          key = tuple(tokens[i:i+1])
        #         else: 
        #             key = tuple(tokens[i-j:i+1]) if i>j and j>0 else tuple(tokens[0:(0+ntemp)])
        #         print "key is {}, iterater is {}, ntemp is {}".format(key, i, ntemp)
        #             unk = '<unk_{}>'.format(ntemp - 1)
        #             if key not in self.nprob_dic[ntemp]:
        #                 key = tuple([unk, tokens[i+ntemp-1]])
        #                 print "Fixed key is {}, iterater is {}, ntemp is {}".format(key, i, ntemp)

        #                 if j == 1:
        #                     bi_fix += 1
        #                 else:
        #                     tri_fix += 1
        #         prob_element = self.nprob_dic[ntemp][key]
        #         prob_tup.append(prob_element)
        #     prob_tup = np.array(prob_tup)
        #     prob = prob_tup.dot(self.r)

        #     perp -= log(prob)
        # print "there are {} fixes for bigram, {} fixes for trigram".format(bi_fix,tri_fix)

        perp = exp(1.0 * perp / len(tokens))
        return perp