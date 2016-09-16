# goot-turing ngram

import os
import random
import operator
import preprocess
from math import pow

c_max = 5

class gt_ngram:
    def __init__(self, content):
        self.nprob_dic, self.nhash_dic, self.ncounter_dic = {}, {}, {}
        self.content = content

    def ntoken_count(self, n):
        tokens = self.content.split()
        counter = {}

        if n == 1:
            unk_set = set()
            # count tokens and put token only apears once into unk_set
            for key in tokens:
                if key in unk_set:
                    counter[key] = 2
                    unk_set.remove(key)
                elif key in counter:
                    counter[key] += 1
                else:
                    unk_set.add(key)
            counter['<unk>'] = len(unk_set)

            # update content: replace token only apears once to 'unk'
            for i in xrange(len(tokens)):
                if tokens[i] in unk_set:
                    tokens[i] = '<unk>'
            self.content = ' '.join(tokens)

        else:
            # initialize the counter with keys and value = 0
            _len = len(tokens)
            for i in xrange(_len - n + 1):
                key = tuple(tokens[i:(i + n)])
                counter[key] = counter.get(key, 0) + 1

            # Good-Turing counts
            c_dict = {0: [tuple(['<unk>'])]}
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

    def ngram_generator(self, n, content):
        pass

    def perplexity(self, n, sentences):
        pass

def main():
    indir = "/Users/Christina/DropBox/Courses/CS4740/cs4740/pro1/data/classification_task/atheism/train_docs"
    content = preprocess.preprocess(indir)
    atheism = gt_ngram(content)
    print atheism.ntoken_count(2)

if __name__ == "__main__":
    main()
