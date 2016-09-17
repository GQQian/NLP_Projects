# goot-turing ngram

import os
import random
import operator
import preprocess
from math import pow

c_max = 5

class gt_ngram(object):
    def __init__(self, content):
        self.nprob_dic, self.nhash_dic, self.ncounter_dic = {}, {}, {}
        self.content = content


    def ntoken_count(self, n):
        tokens = self.content.split()


        if n == 1 or 1 not in self.ncounter_dic:
            counter = {}
            unk_set = set()
            # count tokens and put token apears for the first time into unk_set
            for key in tokens:
                key = tuple([key])
                if key in counter:
                    counter[key] += 1
                elif key in unk_set:
                    counter[key] = 1
                else:
                    unk_set.add(key)
            counter[tuple(['<unk_1>'])] = len(unk_set)

            # update content: replace token apears for the first time to '<unk_1>'
            for i in xrange(len(tokens)):
                if tuple([tokens[i]]) in unk_set:
                    unk_set.remove(tuple([tokens[i]]))
                    tokens[i] = '<unk_1>'
            self.content = ' '.join(tokens)
            self.ncounter_dic[1] = counter

        if n > 1:
            counter = {}
            # initialize the counter with keys and value = 0
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

                num2 = len(c_dict[c + 1])
                # TODO what if num1 == 0? Or just forget this situation as it little happens
                c_new = 1.0 * (c + 1) * num2 / num1
                counter.update(dict((tokens, c_new) for tokens in _list))

            self.ncounter_dic[n] = counter

        return counter


    def prob_generator(self, n):
        self.ncounter_dic[n] = self.ncounter_dic[n] if n in self.ncounter_dic else self.ntoken_count(n)
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


    def perplexity(self, n, sentences):
        pass

def main():
    indir = "/Users/Christina/DropBox/Courses/CS4740/cs4740/pro1/data/classification_task/atheism/train_docs"
    content = preprocess.preprocess(indir)
    atheism = gt_ngram(content)
    print atheism.prob_generator(3)

if __name__ == "__main__":
    main()
