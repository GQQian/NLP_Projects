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
        perp = 0
        for i, token in enumerate(tokens):
            key = tuple([token])
            if key not in self.ncounter_dic[1]:
                tokens[i] = '<unk_1>'

            _len = len(tokens)

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
                # print "key is {}, iterater is {}, j is {}, n is {}, dict_len is {}".format(key, i, j, n, self.nprob_dic.keys())

                if key == () or (('<s>' in key) and key[-1] != '<s>' and j>0):
                    prob_element = 0
                else:
                    prob_element = self.nprob_dic[j+1][key]
                # prob_element = self.nprob_dic[j+1][key] if (key != () or !((() in key) and key(2) != ())) else 0
                prob_tup.append(prob_element)
            prob_tup = np.array(prob_tup)
            prob = prob_tup.dot(r)
            perp -= log(prob)

        perp = exp(1.0 * perp / len(tokens))
        # print "there are {} <unk> for bigram, {} <unk> for trigram, with a total of {} tokens".format(bi_fix, tri_fix, iters[0])
        return perp









        # for z in xrange(3):
        #     x = z + 1
        #     self.nprob_dic[x] = self.nprob_dic[x] if x in self.nprob_dic else self.generate_ngram(x)
        # tokens = preprocess.preprocess_text(sentences)
        # # tokens = preprocess.preprocess_text(sentences).split()
        # # Prepare sentences for each ngram
        # token_list = [[],[],[]]
        # token_list[0] = tokens.replace('<s>', '').split()
        # token_list[1] = tokens.split()
        # token_list[2] = tokens.replace('<s>', '<s1> <s2>').split()
        # tokens = tokens.split()
        # # for keys in self.nprob_dic.keys():
        # # use unk_1 to repalce word not in ncounter_dic[1]
        # self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
        # for eachlist in token_list:
        #     for i, token in enumerate(eachlist):
        #         key = tuple([token])
        #         if key not in self.ncounter_dic[1]:
        #             eachlist[i] = '<unk_1>'
        #
        # # calculate perplexity
        # perp = 0
        #
        # _len = len(token_list[0])
        # bi_fix = 0  # count <unk> in bigram
        # tri_fix = 0 # count <unk> in trigram
        #
        #
        # iters = [0, 0, 0]
        #
        # while iters[0] < _len - 1:
        #     prob = 0
        #     prob_tup = []
        #     for j in xrange(n):
        #         if j < 1:
        #             key = tuple(token_list[j][iters[j]: iters[j] + 1])
        #             while '</s>' in key:
        #
        #                 key = tuple(token_list[j][iters[j]: iters[j] + 1])
        #                 iters[j] += 1
        #             iters[j] += 1
        #
        #         else:
        #
        #             key = tuple(token_list[j][iters[j]: iters[j] + j + 1])
        #
        #             while '</s>' in key:
        #
        #                 key = tuple(token_list[j][iters[j] : iters[j] + j + 1])
        #                 iters[j] += 1
        #             # print "key is {}, iterater is {}, j is {}".format(key, iters[0], j)
        #
        #             # Replace unknown word with <unk> #
        #             unk = '<unk_{}>'.format(j)
        #             if key not in self.nprob_dic[j + 1]:
        #                 key = tuple([unk, tokens[-1]])
        #             # print "key is {}, iterater is {}, j is {}".format(key, iters[0], j)
        #                 if j == 1:
        #                     bi_fix += 1
        #                 else:
        #                     tri_fix += 1
        #             iters[j] += 1
        #         prob_element = self.nprob_dic[j+1][key]
        #         prob_tup.append(prob_element)
        #     prob_tup = np.array(prob_tup)
        #     prob = prob_tup.dot(r)
        #     perp -= log(prob)
        # perp = exp(1.0 * perp / len(tokens))
        # # print "there are {} <unk> for bigram, {} <unk> for trigram, with a total of {} tokens".format(bi_fix, tri_fix, iters[0])
        # return perp
