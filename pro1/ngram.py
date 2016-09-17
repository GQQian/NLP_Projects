import os
import random
import operator

class ngram(object):
    def __init__(self):
        """
        Parameter:
        ncounter_dic: stores the counts for grams, ncounter_dic[1] stores grams for unigram, 
            ncounter_dic[2][(key1,key2)] will return the count for (key1,key2) in bigram dictionary;
        nprob_dic: stores the counts for grams, nprob_dic[1] stores grams for unigram, 
            nprob_dic[2][(key1,key2)] will return the probability for (key1,key2) in bigram dictionary;
        
        """

        self.nprob_dic, self.nhash_dic, self.ncounter_dic = {}, {}, {}


    def ntoken_count(self,n, content):
        """ 
        Input: n: when n = 1,2,3 indicates unigram, bigram and trigram respectively;
                   content: list of input sentences
        Output counter: dictionary, with key, value indicates n-gram tuples, counts respectively
        """

        counter = {}
        tokens = content.split()
        _len = len(tokens)
        for i in xrange(_len - n + 1):
            key = tuple(tokens[i:(i + n)])
            counter[key] = counter.get(key, 0) + 1
        return counter


    def generate_ngram(self,n, content):
        """ 
        Input: n: when n = 1,2,3 indicates unigram, bigram and trigram respectively;
               content: list of input sentences
        Output self.nprob_dic[n]: dictionary, with key, value indicates n-gram tuples, probability respectively
        """

        self.ncounter_dic[n] = self.ncounter_dic[n] if n in self.ncounter_dic else self.ntoken_count(n, content)
        self.nhash_dic[n], self.nprob_dic[n] = {}, {}

        if n == 1:
            _sum = sum(self.ncounter_dic[n].values())
            self.nprob_dic[n] = dict((key, num * 1.0 / _sum) for key, num in self.ncounter_dic[n].items())
        elif n > 1:
            self.ncounter_dic[n - 1] = self.ncounter_dic[n - 1] if n - 1 in self.ncounter_dic else self.ntoken_count(n - 1, content)
            for key_n, num_n in self.ncounter_dic[n].items():
                key_nminus1 = key_n[:-1]

                self.nhash_dic[n][key_nminus1] = self.nhash_dic[n].get(key_nminus1, [])
                self.nhash_dic[n][key_nminus1].append(key_n)

                num_nminus1 = self.ncounter_dic[n - 1][key_nminus1]
                self.nprob_dic[n][key_n] = 1.0 * num_n / num_nminus1

        return self.nprob_dic[n]


    def generate_sentence(self,n, content, sentence = '<s>'):
        """ 
        Input: n: when n = 1,2,3 indicates unigram, bigram and trigram respectively;
               content: list of input sentences

        Output: normalized_sentence: random sentence when sentence is not set, complete the sentence if
                sentence is set to some starting words
        """

        sentence_minlen = 5
        sentence_maxlen = 100
 
        # if do not have n-gram result, then choose (n-1)-gram
        def produce_next(n, content, sentence_list):
            self.nprob_dic[n] = self.nprob_dic[n] if n in self.nprob_dic else self.generate_ngram(n, content)
            rand_prob = random.uniform(0.0, 1.0)
            prob_sum = 0

            if n == 1:
                for token in self.nprob_dic[1]:
                    prob_sum += self.nprob_dic[1][token]
                    if prob_sum > rand_prob:
                        sentence_list.append(token[0])
                        break
            else:
                key = tuple(sentence_list[-n+1:])
                if key not in self.nhash_dic[n]:
                    sentence_list = produce_next(n - 1, content, sentence_list)
                else:
                    for token in self.nhash_dic[n][key]:
                        prob_sum += self.nprob_dic[n][tuple(token)]
                        if prob_sum > rand_prob:
                            sentence_list.append(token[-1])
                            break

            return sentence_list

        normalized_sentence = ""
        if not sentence.startswith('<s> '):
            sentence = '<s> ' + sentence

        while len(normalized_sentence) < sentence_minlen:
            sentence_list = sentence.split()
            while len(sentence_list) < sentence_minlen or \
                (len(sentence_list) < sentence_maxlen and sentence_list[-1] != '</s>'):
                sentence_list = produce_next(min(len(sentence_list) + 1, n), content, sentence_list)

            # normalize: remove '<s>', '<\s>', multiple white spaces
            normalized_sentence = ' '.join(sentence_list).replace('<s>', '').replace('</s>', '')
            normalized_sentence = ' '.join(normalized_sentence.split())

        normalized_sentence = normalized_sentence[0].upper() + normalized_sentence[1:]

        return normalized_sentence
