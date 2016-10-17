# linear interpolation ngram
import os
import random
from gt_ngram import gt_ngram
import operator
import preprocess
from math import pow, log, exp
import numpy as np

class li_ngram(gt_ngram):
    def __init__(self, content):
        super(li_ngram, self).__init__(content)


    def generate_perplexity(self, n, sentences, r = [0.05,0.25,0.7]):
        if n == 1:
            return super(li_ngram, self).generate_perplexity(n, sentences)

        for z in xrange(n):
            x = z + 1
            self.nprob_dic[x] = self.nprob_dic[x] if x in self.nprob_dic else self.generate_ngram(x)
        tokens = preprocess.preprocess_text(sentences).split()

        # use unk_1 to repalce word not in ncounter_dic[1]
        self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
        for i, token in enumerate(tokens):
            key = tuple([token])
            if key not in self.ncounter_dic[1]:
                tokens[i] = '<unk_1>'

        _len = len(tokens)
        perp = 0
        for i in xrange(_len):
            prob_tup = []
            for j in xrange(n):
                key = tuple(tokens[i - j : i + 1])

                # accessing key
                if j > 0:
                    # Replace unknown word with <unk> #
                    unk = '<unk_{}>'.format(j)
                    if key != ():
                        if key not in self.nprob_dic[j + 1]:
                            key = tuple([unk, tokens[-1]])

                if key == () or (('<s>' in key) and key[-1] != '<s>' and j>0):
                    prob_element = 0
                else:
                    prob_element = self.nprob_dic[j+1][key]
                prob_tup.append(prob_element)
            prob_tup = np.array(prob_tup)
            prob = prob_tup.dot(r)
            if prob == 0:
                continue
            perp -= log(prob)

        perp = exp(1.0 * perp / len(tokens))
        return perp

        # # calculate perplexity
        # perp = 0
        # # i is the first index of tokens, j is j-gram
        # for i, token in enumerate(tokens):
        #     prob = 0
        #     for j in xrange(1, min(i + 2, n + 1)):
        #         if r[j - 1] < 0.0000001:
        #             continue
        #         if j == 1:
        #             for token in tokens:
        #                 key = tuple([token])
        #                 prob += r[j - 1] * self.nprob_dic[1][key]
        #         else:
        #             unk = '<unk_{}>'.format(j - 1)
        #             if i + j > len(tokens):
        #                 continue
        #             key = tuple(tokens[i:(i + j)])
        #             if key not in self.nprob_dic[n]:
        #                 key = tuple([unk, tokens[i + j - 1]])
        #                 if key not in self.nprob_dic[j]:
        #                     key = tuple([unk, '<unk_1>'])
        #             prob += r[j - 1] * self.nprob_dic[j][key]
        #
        #             # print "part: {}".format(r[j - 1] * self.nprob_dic[j][key])
        #             # print "sum: {}".format(prob)
        #
        #         if prob != 0:
        #             perp -= log(prob)
        #
        # perp = exp(1.0 * perp / len(tokens))
        # return perp
