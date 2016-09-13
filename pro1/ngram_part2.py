import os
import nltk
import random
import re
import operator
import itertools
from nltk.tokenize import sent_tokenize, word_tokenize

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sentence_maxlen = 100
sentence_minlen = 5
gt_max = 5
nprob_dic, nhash_dic, ncounter_dic, gt_ncounter_dic = {}, {}, {}, {}

def preprocess(indir):
    def remove_punctuation(text):
        text = text.replace('_', '')
        result = re.findall(r'[\w\,\.\!\?]+',text)
        return ' '.join(result)

    def remove_email(text):
        result = re.sub(r'[\w\.-]+@[\w\.-]+','',text)
        return result


    buffer, output = "", ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            buffer += raw_content

    # normalize
    buffer = buffer.lower()
    buffer = remove_email(buffer)
    buffer = remove_punctuation(buffer)

    # corner case
    buffer = buffer.replace(' i ', ' I ')
    buffer = buffer.replace(' i\' ', ' I\' ')

    sent_list = sent_tokenize(buffer)
    for sent in sent_list:
        output += " <s> " + sent + " </s> "

    return output


def ntoken_count(n, content):
    ncounter_dic[n] = {}
    tokens = content.split()
    _len = len(tokens)
    for i in xrange(_len - n + 1):
        key = tuple(tokens[i:(i + n)])
        ncounter_dic[n][key] = ncounter_dic[n].get(key, 0) + 1
    return ncounter_dic[n]


def gt_ntoken_count(n, content):
    if n == 1:
        return ncounter_dic[1] if 1 in ncounter_dic else ntoken_count(1, content)

    gt_ncounter_dic[n] = {}
    ncounter_dic[n] = ncounter_dic[n] if n in ncounter_dic else ntoken_count(n, content)
    ncounter_dic[1] = ncounter_dic[1] if n in ncounter_dic else ntoken_count(1, content)

    for tokens in itertools.product(ncounter_dic[1].keys(), repeat = n)):
        tokens = tuple(tokens)

        # copy ncounter_dic into gt_ncounter_dic, add 0 count tokens
        if tokens not in ncounter_dic[n]:
            gt_ncounter_dic[n][tokens] = 0
        else:
            gt_ncounter_dic[n][tokens] = ncounter_dic[n][tokens]

        num = gt_ncounter_dic[n][tokens]
        if num < gt_max:
            if num in c_dict:
                c_dict[num].append(tokens)
            else
                c_dict[num] = [tokens]



    for num, _list in c_dict.items():
        if num + 1 not in c_dict:
            gt_ncounter_dic[n].update(dict(tokens, num) for tokens in _list)
        else:
            _sum = sum(_list)
            _sum1 = sum(c_dict(num + 1))




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
    normalized_sentence = ""
    if not sentence.startswith('<s> '):
        sentence = '<s> ' + sentence

    while len(normalized_sentence) < sentence_minlen:
        sentence_list = sentence.split()
        while len(sentence_list) < sentence_minlen or \
            (len(sentence_list) < sentence_maxlen and sentence_list[-1] != '</s>'):
            sentence_list = backoff_produce_next(min(len(sentence_list) + 1, n), content, sentence_list)

        # normalize: remove '<s>', '<\s>', multiple white spaces
        normalized_sentence = ' '.join(sentence_list).replace('<s>', '').replace('</s>', '')
        normalized_sentence = ' '.join(normalized_sentence.split())

    normalized_sentence = normalized_sentence[0].upper() + normalized_sentence[1:]

    return normalized_sentence


def backoff_produce_next(n, content, sentence_list):
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
            sentence_list = backoff_produce_next(n - 1, content, sentence_list)
        else:
            for token in nhash_dic[n][key]:
                prob_sum += nprob_dic[n][tuple(token)]
                if prob_sum > rand_prob:
                    sentence_list.append(token[-1])
                    break

    return sentence_list


def interpolation_produce_next(n, content, sentence_list, r):
    mix_prob_dict = interpolation_prob(n, content, sentence_list, r)
    rand_prob = random.uniform(0.0, 1.0)
    prob_sum = 0
    for token in mix_prob_dict:
        prob_sum += mix_prob_dict[token]
        if prob_sum > rand_prob:
            sentence_list.append(token)

    return sentence_list


def interpolation_prob(n, content, sentence_list, r):
    for i in xrange(1, 4):
        nprob_dic[i] = nprob_dic[i] if i in nprob_dic else ngram_generator(i, content)

        if i == 1:
            mix_prob_dict = dict((key[0], val * r[1]) for key, val in nprob_dic[1].items())
        else:
            if len(sentence_list) < i - 1:
                break;
            key = tuple(sentence_list[-i+1:])
            if key in nhash_dic[i]:
                for token in nhash_dic[n][key]:
                    mix_prob_dict[token[-1]] += r[i] * nprob_dic[i][tuple(token)]

    # normalize the mix_prob_dict
    _sum = sum(mix_prob_dict.values())
    mix_prob_dict = dict((key, val / _sum) for key in mix_prob_dict)
    return mix_prob_dict


def perplexity(n, content, sentence, r):
    sentence = " <s> " + sentence_list + " </s> "
    sentence_list = sentence.split()
    _sum = 0

    for i in xrange(sentence_list):
        token = sentence_list[i]
        mix_prob_dict = interpolation_prob(n, content, sentence_list[:i], r)
        prob = mix_prob_dict[token]
        _sum += math.log(prob)

    return exp(_sum)


def main():
    # TODO: input
    argv = ["data/autos/train_docs", "5", "I have"]

    # check input
    if (len(argv) == 0):
        print "Please input a topic"

    topic = argv[0]
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() \
        and int(argv[1]) >= 1 else 1
    sent_pre = argv[2] if len(argv) > 2 else ""

    indir, outdir = indir_pre + topic, outdir_pre + topic

    if not os.path.isdir(indir):
        print "Sorry, the topic does not exist!"
        return
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # preprocess
    content = preprocess(indir)

    # print sentence generator
    for k in xrange(1, n + 1):
        print "\n\n[{}-gram]\n".format(k)

        print "Empty sentence"
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + sentence_generator(k, content)

        print "\nWith incomplete sentence: " + "\"{}\"".format(sent_pre)
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + sentence_generator(k, content, sent_pre)


if __name__ == "__main__":
    main()
