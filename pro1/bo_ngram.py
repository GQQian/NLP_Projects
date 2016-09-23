# linear interpolation ngram
import os
import random
from gt_ngram import gt_ngram
import operator
import preprocess
from math import pow, log, exp
import numpy as np

class bo_ngram(gt_ngram):
    def __init__(self, content):
        super(bo_ngram, self).__init__(content)
        # r[0] for bigram, r[1] for unigram



    def generate_perplexity(self, n, sentences, r = [0.4, 0.6, 1]):
        for z in xrange(n):
            x = z + 1
            self.nprob_dic[x] = self.nprob_dic[x] if x in self.nprob_dic else self.generate_ngram(x)
        tokens = preprocess.preprocess_text(sentences).split()

        # Prepare sentences for each ngram
        # token_list = [[],[],[]]
        # token_list[0] = tokens.replace('<s>', '').split()
        # token_list[1] = tokens.split()
        # token_list[2] = tokens.replace('<s>', '<s1> <s2>').split()
        # tokens = tokens.split()

        # use unk_1 to repalce word not in ncounter_dic[1]
        self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
        for i, token in enumerate(tokens):
            key = tuple([token])
            if key not in self.ncounter_dic[1]:
                tokens[i] = '<unk_1>'

        # calculate perplexity 
        perp = 0 

        _len = len(tokens)

        # iters = [0, 0, 0]

        for i in xrange(_len):
            prob_tup = []
            for j in xrange(n):
                key = tuple(tokens[i - j : i + 1])

                if j > 0:
                    unk = '<unk_{}>'.format(j)
                    if key != ():
                        if key not in self.nprob_dic[j + 1]:
                            key = tuple([unk, tokens[-1]])
                if key == () or (('<s>' in key) and key[-1] != '<s>' and j>0):
                    prob_element = 0
                else:
                    prob_element = self.nprob_dic[j+1][key]
                prob_tup.append(prob_element)
            ntemp = n-1
            while(prob_tup[ntemp] == 0 or prob_tup[ntemp] == 1):
                ntemp -= 1

            prob = prob_tup[ntemp] * r[ntemp]

            perp -= log(prob)
        perp = exp(1.0 * perp / len(tokens))
        return perp
        # while iters[0] < _len - 1:
        #     prob = 0
        #     prob_tup = []
        #     for j in xrange(n):
        #         key = tuple(tokens[i - j : i + 1])
        #         if j > 0:
        #             key = tuple(token_list[j][iters[j]: iters[j] + 1])
        #             # while '</s>' in key:
        #             #     key = tuple(token_list[j][iters[j]: iters[j] + 1])
        #             #     iters[j] += 1
        #             # iters[j] += 1
        #             prob_element = self.nprob_dic[j+1][key]
        #             prob_tup.append(prob_element)
        #         else:
        #             key = tuple(token_list[j][iters[j]: iters[j] + j + 1])
        #             while '</s>' in key:
        #                 key = tuple(token_list[j][iters[j] : iters[j] + j + 1])
        #                 iters[j] += 1
        #             # Replace unknown word with <unk> #
        #             unk = '<unk_{}>'.format(j)
        #             if key not in self.nprob_dic[j + 1]:
        #                 prob_element = 1
        #             else:
        #                 prob_element = self.nprob_dic[j+1][key]
        #             prob_tup.append(prob_element)
        #             iters[j] += 1