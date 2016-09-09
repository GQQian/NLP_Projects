import os
import random

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sentence_maxlen = 100
sentence_minlen = 3
nprob_dic, nhash_dic, ncounter = {}, {}, {}

def preprocess(indir):
    # TODO
    # email, Upper-lower case
    # sentence boundary
    # ...

    pro_content = ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            pro_content += raw_content

    return pro_content


def ntoken_count(n, content):
    counter = {}
    tokens = content.split()
    _len = len(tokens)
    for i in xrange(_len - n + 1):
        key = tuple(tokens[i:(i + n)])
        counter[key] = counter.get(key, 0) + 1

    return counter


def ngram_generator(n, content):
    ncounter[n] = ncounter[n] if n in ncounter else ntoken_count(n, content)
    nhash_dic[n], nprob_dic[n] = {}, {}

    if n == 1:
        _sum = sum(ncounter[n].values())
        nprob_dic[n] = dict((key, num * 1.0 / _sum) for key, num in ncounter[n].items())
    elif n > 1:
        ncounter[n - 1] = ncounter[n - 1] if n - 1 in ncounter else ntoken_count(n - 1, content)
        for key_n, num_n in ncounter[n].items():
            key_nminus1 = key_n[:-1]

            nhash_dic[n][key_nminus1] = nhash_dic[n].get(key_nminus1, [])
            nhash_dic[n][key_nminus1].append(key_n)

            num_nminus1 = ncounter[n - 1][key_nminus1]
            nprob_dic[n][key_n] = 1.0 * num_n / num_nminus1
    return nprob_dic[n]


def sentence_generator(n, content, sentence = '<s>'):
    def produce_next_token(n, content, sentence_list):
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
            key = tuple(sentence_list[-n:])
            if key not in nhash_dic[n]:
                sentence_list = produce_next_token(n - 1, content, sentence_list)
            else:
                for token in nhash_dic[n][key]:
                    prob_sum += nprob_dic[n][tuple(token)]
                    if prob_sum > rand_prob:
                        sentence_list.append(token[0])
                        break

        return sentence_list

    normalized_sentence = ""
    if not sentence.startswith('<s> '):
        sentence = '<s> ' + sentence

    while not len(normalized_sentence):
        sentence_list = sentence.split()
        while len(sentence_list) < sentence_minlen or \
            (len(sentence_list) < sentence_maxlen and sentence_list[-1] != '</s>'):
            sentence_list = produce_next_token(min(len(sentence_list) + 1, n), content, sentence_list)

        # normalize the sentence
        normalized_sentence = ' '.join(sentence_list).replace('<s>', '').replace('</s>', '')
        normalized_sentence = ' '.join(normalized_sentence.split())


    normalized_sentence = normalized_sentence[0].upper() + normalized_sentence[1:]

    return normalized_sentence


def main():
    argv = ["data/autos/train_docs", "3"] # TODO: input
    # argv = ["test", "2"] # TODO: input
    if (len(argv) == 0):
        print "Please input a topic"

    topic = argv[0]
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() \
        and int(argv[1]) >= 1 else 1

    indir, outdir = indir_pre + topic, outdir_pre + topic

    if not os.path.isdir(indir):
        print "Sorry, the topic does not exist!"
        return
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    content = preprocess(indir)

    for k in xrange(1, n + 1):
        print "\n\n[{}-gram]\n".format(k)
        for i in xrange(2):
            print sentence_generator(n, content) + "\n"


if __name__ == "__main__":
    main()
