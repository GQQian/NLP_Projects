# goot-turing ngram

import config
import os
import random
import operator

nprob_dic, nhash_dic, ncounter_dic = {}, {}, {}
def ntoken_count(n, content):
    tokens = content.split()

    if n == 1:
        unk_set = Set()
        counter = {}
        for token in tokens:
            if token in unk_set:
                counter[token] = 2
            elif token in counter:
                counter[token] += 1
            else:
                unk_set.add(token)
        counter['unk'] = len(unk_set)
    else:
        # _len = len(tokens)
        # for i in xrange(_len - n + 1):
        #     key = tuple(tokens[i:(i + n)])
        #     counter[key] = counter.get(key, 0) + 1
        # return counter
        pass
        # TODO: Unfinished
