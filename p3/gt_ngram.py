# goot-turing ngram

import os
import random
import operator
from math import pow, log, exp
import sys
c_max = 5

class gt_ngram(object):
    def __init__(self, content):
        self.nprob_dic, self.nhash_dic, self.ncounter_dic = {}, {}, {}

        # use unk_1 to replace the first appeared word
        tokens = content.split()
        _set = set()
        for i, token in enumerate(tokens):
            if token not in _set:
                tokens[i] = '<unk_1>'
                _set.add(token)

        self.content = content + ' '.join(tokens)


    def ntoken_count(self, n):
        counter = {}
        tokens = self.content.split()
        if n == 1:
            for key in tokens:
                key = tuple([key])
                counter[key] = counter.get(key, 0) + 1
        else:
            _len = len(tokens)
            for i in xrange(_len - n + 1):
                key = tuple(tokens[i:(i + n)])
                counter[key] = counter.get(key, 0) + 1

            # Good-Turing counts

            # if a ngram not exists, we use ['<unk_n>'] to replace it,
            # and the initial count for it is 0
            c_dict = {0: [tuple(['<unk_{}>'.format(n)])]}
            for key, c in counter.items():
                if c > c_max:
                    continue
                if c not in c_dict:
                    c_dict[c] = [key]
                else:
                    c_dict[c].append(key)

            for c, _list in c_dict.items():
                if c == c_max:
                    continue

                if c == 0:
                    self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
                    num1 = pow(len(self.ncounter_dic[1].keys()), n) - len(counter.keys())

                else:
                    num1 = len(_list)

                if c + 1 not in c_dict:
                    continue
                num2 = len(c_dict[c + 1])
                #if(num1 != 0):
                c_new = 1.0 * (c + 1) * num2 / num1
                #else:
                #    c_new = 1.0 * (c + 1) * num2

                counter.update(dict((tokens, c_new) for tokens in _list))

        self.ncounter_dic[n] = counter

        return counter


    def generate_ngram(self, n):
        self.ncounter_dic[n] = self.ncounter_dic[n] if n in self.ncounter_dic else self.ntoken_count(n)
        # TODO: jiaojiao: check if nhash_dic needed?
        self.nhash_dic[n], self.nprob_dic[n] = {}, {}

        if n == 1:
            _sum = sum(self.ncounter_dic[n].values())
            self.nprob_dic[n] = dict((key, num * 1.0 / _sum) for key, num in self.ncounter_dic[n].items())
        elif n > 1:
            self.ncounter_dic[n - 1] = self.ncounter_dic[n - 1] if n - 1 in self.ncounter_dic else self.ntoken_count(n - 1)
            unk_nminus1 = tuple(['<unk_{}>'.format(n - 1)])
            unk_n = tuple(['<unk_{}>'.format(n)])
            self.nhash_dic[n][unk_nminus1] = []
            for key_n, num_n in self.ncounter_dic[n].items():
                if key_n == unk_n:
                    continue
                temp = tuple([unk_nminus1[0], key_n[-1]])
                if n > 2 and temp not in self.nprob_dic[n]:
                    self.nhash_dic[n][unk_nminus1] = temp
                    num_nminus1 = self.ncounter_dic[n - 1][unk_nminus1]
                    self.nprob_dic[n][temp] = 1.0 * self.ncounter_dic[n][unk_n] / num_nminus1

                key_nminus1 = key_n[:-1]
                self.nhash_dic[n][key_nminus1] = self.nhash_dic[n].get(key_nminus1, [])
                self.nhash_dic[n][key_nminus1].append(key_n)

                num_nminus1 = self.ncounter_dic[n - 1][key_nminus1]
                self.nprob_dic[n][key_n] = 1.0 * num_n / num_nminus1

        return self.nprob_dic[n]


    def generate_perplexity(self, n, sentences):
        self.nprob_dic[n] = self.nprob_dic[n] if n in self.nprob_dic else self.generate_ngram(n)
        tokens = sentences.split()

        # use unk_1 to repalce word not in ncounter_dic[1]
        self.ncounter_dic[1] = self.ncounter_dic[1] if 1 in self.ncounter_dic else self.ntoken_count(1)
        for i, token in enumerate(tokens):
            key = tuple([token])
            if key not in self.ncounter_dic[1]:
                tokens[i] = '<unk_1>'

        # calculate perplexity
        perp = 0
        if n == 1:
            for token in tokens:
                key = tuple([token])
                prob = self.nprob_dic[1][key]
                perp -= log(prob)
        else:
            unk = '<unk_{}>'.format(n - 1)
            _len = len(tokens)
            for i in xrange(_len - n + 1):
                key = tuple(tokens[i:(i + n)])
                if key not in self.nprob_dic[n]:
                    key = tuple([unk, tokens[i + n - 1]])
                    if key not in self.nprob_dic[n]:
                        key = tuple([unk, '<unk_1>'])

                prob = self.nprob_dic[n][key]
                perp -= log(prob)
        if(len(tokens) == 0):
            perp = sys.maxint
        else:
            perp = exp(1.0 * perp / len(tokens))
        return perp

"""
def main():
    indir = "/Users/Christina/DropBox/Courses/CS4740/cs4740/pro1/data/classification_task/atheism/train_docs"
    test_f = "/Users/Christina/DropBox/Courses/CS4740/cs4740/pro1/data/classification_task/test_for_classification/file_0.txt"
    content = preprocess.preprocess_dir(indir)
    atheism = gt_ngram(content)


if __name__ == "__main__":
    main()
"""