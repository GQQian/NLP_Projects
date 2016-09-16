import os
import random
import operator

# TODO lili: convert to class
nprob_dic, nhash_dic, ncounter_dic = {}, {}, {}
def ntoken_count(n, content):
    counter = {}
    tokens = content.split()
    _len = len(tokens)
    for i in xrange(_len - n + 1):
        key = tuple(tokens[i:(i + n)])
        counter[key] = counter.get(key, 0) + 1
    return counter


def ngram_generator(n, content):
    ncounter_dic[n] = ncounter_dic[n] if n in ncounter_dic else ntoken_count(n, content)
    nhash_dic[n], nprob_dic[n] = {}, {}

    if n == 1:
        _sum = sum(ncounter_dic[n].values())
        nprob_dic[n] = dict((key, num * 1.0 / _sum) for key, num in ncounter_dic[n].items())
    elif n > 1:
        ncounter_dic[n - 1] = ncounter_dic[n - 1] if n - 1 in ncounter_dic else ntoken_count(n - 1, content)
        for key_n, num_n in ncounter_dic[n].items():
            key_nminus1 = key_n[:-1]

            nhash_dic[n][key_nminus1] = nhash_dic[n].get(key_nminus1, [])
            nhash_dic[n][key_nminus1].append(key_n)

            num_nminus1 = ncounter_dic[n - 1][key_nminus1]
            nprob_dic[n][key_n] = 1.0 * num_n / num_nminus1

    return nprob_dic[n]


def sentence_generator(n, content, sentence = '<s>'):
    sentence_minlen = 5
    sentence_maxlen = 100

    # if do not have n-gram result, then choose (n-1)-gram
    def produce_next(n, content, sentence_list):
        nprob_dic[n] = nprob_dic[n] if n in nprob_dic else ngram_generator(n, content)
        rand_prob = random.uniform(0.0, 1.0)
        prob_sum = 0

        if n == 1:
            for token in nprob_dic[1]:
                prob_sum += nprob_dic[1][token]
                if prob_sum > rand_prob:
                    sentence_list.append(token[0])
                    break
        else:
            key = tuple(sentence_list[-n+1:])
            if key not in nhash_dic[n]:
                sentence_list = produce_next(n - 1, content, sentence_list)
            else:
                for token in nhash_dic[n][key]:
                    prob_sum += nprob_dic[n][tuple(token)]
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
