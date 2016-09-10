import os
import nltk
import random
import re
import operator
from nltk.tokenize import sent_tokenize, word_tokenize

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sentence_maxlen = 100
sentence_minlen = 5
nprob_dic, nhash_dic, ncounter_dic = {}, {}, {}

def preprocess(indir):
    # TODO
    # Upper-lower case
    # temp = sent_tokenize(indir)
    # output = ""
    # delete email ?
    # ...
    def remove_punctuation(text):
    #     pat = re.compile(r"\p{P}+")
        result = re.findall(r'[\w]+',text)
        delim = " "
        return delim.join(result)

    buffer, output = "", ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            buffer += raw_content
    temp = sent_tokenize(buffer)

    for sent in temp:
        output += "<s> " + sent + " </s> "
    # final = remove_punctuation(output)
    final = output
    return final

def preprocess_jiao(indir):
    buffer, output = "", ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            buffer += raw_content

    # normalize
    buffer = buffer.lower()
    buffer = buffer.replace('-', '')
    buffer = buffer.replace('\\', '')
    buffer = buffer.replace('/', '')
    buffer = buffer.replace('_', '')
    buffer = buffer.replace('|', '')
    buffer = buffer.replace('(', '')
    buffer = buffer.replace(')', '')
    buffer = buffer.replace('<', '')
    buffer = buffer.replace('>', '')
    buffer = buffer.replace('|', '')
    buffer = buffer.replace('\"', '')
    buffer = buffer.replace(',', '')
    buffer = buffer.replace('=', '')
    buffer = buffer.replace('#', '')
    buffer = buffer.replace(' i ', ' I ')
    buffer = buffer.replace(' i\' ', ' I\' ')


    temp = sent_tokenize(buffer)

    for sent in temp:
        output += " <s> " + sent + " </s> "

    final = output
    return final


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


def main():
    argv = ["data/autos/train_docs", "5", "I have"] # TODO: input
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

    # content = preprocess(indir)
    content = preprocess_jiao(indir)


    for k in xrange(1, n + 1):
        print "\n\n[{}-gram]\n".format(k)

        print "Empty sentence"
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + sentence_generator(n, content)

        print "\nWith incompelete sentence: " + "\"{}\"".format(sent_pre)
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + sentence_generator(n, content, sent_pre)


if __name__ == "__main__":
    main()
