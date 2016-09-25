from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram
from li_ngram import li_ngram
from bo_ngram import bo_ngram
import sys
import operator
import csv
import numpy as linspace
import numpy
import math
import operator
import re
import copy

# TODO: delete it when not used

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
topics = {'atheism':0, 'autos':1, 'graphics':2, 'medicine':3, 'motorcycles':4, 'religion':5, 'space':6}

def random_sentence_ngram(n = 2, sent_pre = "I have"):
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        ngrams = ngram(content)
        print "\n\n\nTopic: {}\n".format(topic)
        for k in xrange(1, n + 1):
            print "[{}-gram]\n".format(k)

            print "Empty sentence"
            for i in xrange(3):
                print "[{}]  ".format(i + 1) + ngrams.generate_sentence(k)

            print "\nWith incomplete sentence: " + "\"{}\"".format(sent_pre)
            for i in xrange(3):
                print "[{}]  ".format(i + 1) + ngrams.generate_sentence(k, sent_pre)


def generate_perplexity_gt_ngram():
    gt_ngrams = {}
    perplexity = {} # key: filename, value: a dic with (key: n, value: perplexity of ngram)
    for topic in topics:
        print "Topic: {}".format(topic)
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        gt_ngrams[topic] = gt_ngram(preprocess.preprocess_dir(indir))

        # key: n, value: dict(key: filename, value: perplexity of training/test file)
        perp_train, perp_test = {}, {}
        for i in xrange(1, 6):
            # training data
            perp_train[i] = {}
            for root, dirs, filenames in os.walk(indir):
                for f in filenames:
                    content = open(os.path.join(root, f), 'r').read()
                    perp_train[i][f] = gt_ngrams[topic].generate_perplexity(i, content)

            # test data
            perp_test[i] = {}
            test_dir = indir_pre + "data/classification_task/test_for_classification/"
            for root, dirs, filenames in os.walk(test_dir):
                for f in filenames:
                    content = open(os.path.join(root, f), 'r').read()
                    perp_test[i][f] = gt_ngrams[topic].generate_perplexity(i, content)

            # print the average perplexity of training/test data
            print "[{}-gram] training {}".format(i, numpy.mean(perp_train[i].values()))
            print "[{}-gram] test {}".format(i, numpy.mean(perp_test[i].values()))


def topic_classification_gt_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    gt_result.csv in classification_task directory
    """

    # get gt_ngram for each topic and read all test data
    ratio = 0.8
    gt_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        split_train_test(ratio = ratio)
        train_text[topic] = open(train_f, 'r').read()
        gt_ngrams[topic] = gt_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    for i in xrange(1, 10):
        _sum, correct = 0, 0
        for label_topic in topics:
            test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
            for root, dirs, filenames in os.walk(test_dir):
                for idx, f in enumerate(filenames):
                    if idx < len(filenames) * ratio:
                        continue
                    text = open(os.path.join(root, f),'r').read()
                    min_perp, min_topic = sys.maxint, label_topic
                    for topic in topics:
                        perp = gt_ngrams[topic].generate_perplexity(i, text)
                        if perp < min_perp:
                            min_perp = perp
                            min_topic = topic

                    if label_topic == min_topic:
                        correct += 1
                    _sum += 1


        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])

    #choose the best n
    n = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/gt_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def split_train_test(mission = "classification_task", ratio = 0.8):
    """
    split train_docs into     training:test = 4:1
    store the preprocessed file train.txt and test.txt in each topic directory
    """

    # find the nearest </s> after 80% content
    for topic in topics:
        indir = indir_pre + "data/{}/{}/train_docs".format(mission, topic)
        num_file = len(os.listdir(indir))
        num_train_file = math.floor(num_file * ratio)
        train_text, test_text = "", ""

        for root, dirs, filenames in os.walk(indir):
            for i, f in enumerate(filenames):
                raw_content = preprocess.preprocess_file(os.path.join(root, f))
                if i < num_train_file:
                    train_text += raw_content
                else:
                    test_text += raw_content

        train_path = indir_pre + "data/{}/{}/train.txt".format(mission, topic)
        test_path = indir_pre + "data/{}/{}/test.txt".format(mission, topic)
        open(train_path, 'w').write(train_text)
        open(test_path, 'w').write(test_text)



def topic_classification_li_ngram():
    # TODO when li_gram done, test
    # get gt_ngram for each topic and read all test data

    ratio = 0.8
    n = 4
    li_ngrams, train_text = {}, {}#key: topic
    split_train_test(ratio = ratio)
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        train_text[topic] = open(train_f, 'r').read()
        li_ngrams[topic] = li_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy, r = {}, [.004, .015, .03, .9546]
    diff = 0.1
    _sum, correct = 0, 0
    for label_topic in topics:
        test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
        for root, dirs, filenames in os.walk(test_dir):
            for idx, f in enumerate(filenames):
                if idx < len(filenames) * ratio:
                    continue
                text = open(os.path.join(root, f),'r').read()
                min_perp, min_topic = sys.maxint, label_topic
                for topic in topics:
                    perp = li_ngrams[topic].generate_perplexity(n, text, r = r)

                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

    accuracy[tuple(r)] = 1.0 * correct / _sum
    print "{} {}".format(tuple(r), accuracy[tuple(r)])

def topic_classification_bo_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    bo_result.csv in classification_task directory
    """

    # get bo_ngram for each topic and read all test data
    bo_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        bo_ngrams[topic] = bo_ngram(train_text[topic])
    n = 4 
    ratio = .8
    r = [0.001, 0.01, 0.1, 0.89]
    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in bo_ngram
    # r = 
    _sum, correct = 0, 0

    for label_topic in topics:
        test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
        for root, dirs, filenames in os.walk(test_dir):
            for idx, f in enumerate(filenames):
                if idx < len(filenames) * ratio:
                    continue
                text = open(os.path.join(root, f),'r').read()
                min_perp, min_topic = sys.maxint, label_topic
                for topic in topics:
                    perp = bo_ngrams[topic].generate_perplexity(n, text, r = r)

                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

    accuracy[tuple(r)] = 1.0 * correct / _sum
    print "{}: {}".format(r, accuracy[tuple(r)])

def topic_classification_ngram_dis():
    # get gt_ngram for each topic and read all test data
    ratio = 0.8
    ngrams, train_text  = {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        split_train_test(ratio = ratio)
        train_text[topic] = open(train_f, 'r').read()
        ngrams[topic] = ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    max_dis = 500
    for i in xrange(1, 5):
        # rank the counter for each topic
        counter_rank_dic = {}
        for topic in topics:
            counter_rank_dic[topic] = {}
            ngrams[topic].ncounter_dic[i] = ngrams[topic].ncounter_dic[i] if i in ngrams[topic].ncounter_dic else \
                                            ngrams[topic].ntoken_count(i)
            sorted_counter = sorted(ngrams[topic].ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)

            rank_num, num = 1, 0
            for k in sorted_counter:
                if k[1] != num:
                    rank_num += 1
                    num = k[1]
                counter_rank_dic[topic][k[0]] = rank_num

        _sum, correct = 1, 0
        for label_topic in topics:
            test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
            for root, dirs, filenames in os.walk(test_dir):
                for idx, f in enumerate(filenames):
                    if idx < len(filenames) * ratio:
                        continue

                    text = open(os.path.join(root, f),'r').read()
                    test_ngram = ngram(preprocess.preprocess_text(text))
                    test_ngram.ncounter_dic[i] = test_ngram.ncounter_dic[i] if i in test_ngram.ncounter_dic else \
                                                 test_ngram.ntoken_count(i)
                    sorted_counter = sorted(test_ngram.ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)

                    test_counter_rank_list = {}

                    rank_num, num = 1, 0
                    for k in sorted_counter:
                        if k[1] != num:
                            rank_num += 1
                            num = k[1]
                        test_counter_rank_list[k[0]] = rank_num
                    min_dis, min_topic = sys.maxint, label_topic
                    for topic in topics:
                        dis = 0
                        for tokens, num in test_counter_rank_list.items():
                            if tokens in counter_rank_dic[topic]:
                                dis += min(abs(num - counter_rank_dic[topic][tokens]), max_dis)
                            else:
                                dis += max_dis

                        if dis < min_dis:
                            min_dis = dis
                            min_topic = topic

                    if label_topic == min_topic:
                        correct += 1
                    _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])

    #choose the best n
    n = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]
    print n

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/dis_result_1.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = open(os.path.join(root, f),'r').read()
                test_ngram = ngram(preprocess.preprocess_text(text))
                test_ngram.ncounter_dic[i] = test_ngram.ncounter_dic[i] if i in test_ngram.ncounter_dic else \
                                             test_ngram.ntoken_count(i)
                sorted_counter = sorted(test_ngram.ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)

                test_counter_rank_list = {}

                rank_num, num = 1, 0
                for k in sorted_counter:
                    if k[1] != num:
                        rank_num += 1
                        num = k[1]
                    test_counter_rank_list[k[0]] = rank_num
                min_dis, min_topic = sys.maxint, label_topic
                for topic in topics:
                    dis = 0
                    for tokens, num in test_counter_rank_list.items():
                        if tokens in counter_rank_dic[topic]:
                            dis += min(abs(num - counter_rank_dic[topic][tokens]), max_dis)
                        else:
                            dis += max_dis

                    if dis < min_dis:
                        min_dis = dis
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})

def spell_checking_gt_gram():
    f = indir_pre + "data/spell_checking_task/confusion_set.txt"
    lines = [line.rstrip('\n\r') for line in open(f)]
    # key: word   value: a set of confused words
    confusion_dic = {}
    for line in lines:
        if "may" in line:
            continue
        words = line.split()
        for word in words:
            if word in confusion_dic:
                for word1 in confusion_dic[word]:
                    confusion_dic[word1] = confusion_dic[word].union(set(words))
            else:
                confusion_dic[word] = set(words)

    confusion_dic['may'] = set(['may', 'may be'])

    # buid gt_ngram models
    gt_nragms = {}
    ratio = 0.8
    split_train_test(mission = "spell_checking_task", ratio = ratio)
    for topic in topics:
        train_f = indir_pre + "data/spell_checking_task/{}/train.txt".format(topic)
        train_text = open(train_f, 'r').read()
        gt_nragms[topic] = gt_ngram(train_text)

    # get accuracy for i-gram
    accuracy = {}
    for i in xrange(1, 5):
        _sum, correct = 1, 0
        for topic in topics:
            test_dir = indir_pre + "data/spell_checking_task/{}/train_docs".format(topic)
            for root, dirs, filenames in os.walk(test_dir):
                for idx, f in enumerate(filenames):
                    if idx < len(filenames) * ratio:
                        continue
                    text = open(os.path.join(root, f),'r').read()
                    for confusion_word in confusion_dic:
                        indexes = [m.start() for m in  re.finditer(" " + confusion_word + " ", text)]
                        if len(indexes) == 0:
                            continue

                        for index in indexes:
                            sentences = text[max(0, index - 50): min(len(text), index + 50)]
                            left, right = sentences.find(' '), sentences.rfind(' ')
                            sentences = sentences[left : right]
                            curr_index_left = 50 - left
                            curr_index_right = 50 - left + len(confusion_word)

                            min_word, min_perp = confusion_word, sys.maxint
                            for alternative_word in confusion_dic[confusion_word]:
                                alternative_sentences = sentences[0 : curr_index_left] + " " + alternative_word + \
                                                        sentences[curr_index_right + 1:]
                                perp = gt_nragms[topic].generate_perplexity(i, preprocess.preprocess_text(alternative_sentences))

                                if perp < min_perp:
                                    min_perp = perp
                                    min_word = alternative_word

                            _sum += 1
                            if min_word == confusion_word:
                                correct += 1

        accuracy[i] = 1.0 * correct / _sum
        print "{}-gram: {}".format(i, accuracy[i])

    # choose the best n
    n = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]

    # output
    for topic in topics:
        test_dir = indir_pre + "data/spell_checking_task/{}/test_modified_docs".format(topic)
        for root, dirs, filenames in os.walk(test_dir):
            for modified_f in filenames:
                corrected_f =  indir_pre + "data/spell_checking_task/{}/test_docs/".format(topic) +\
                               modified_f.replace('modified', 'corrected')
                text = open(os.path.join(root, modified_f),'r').read()

                for confusion_word in confusion_dic:
                    begin_index = 0
                    while begin_index < len(text) and text[begin_index:].find(" " + confusion_word + " ") >= 0:
                        # find the confusion word
                        replace_left = text[begin_index:].find(" " + confusion_word + " ") + 1 + begin_index
                        replace_right = replace_left + len(confusion_word)

                        # get the content from 50 characters left of the word to 50 characters right of the word
                        sentences = text[max(0, replace_left - 50): min(len(text), replace_left + 50)]
                        left_bound, right_bound = sentences.rfind(' '), sentences.rfind(' ')
                        sentences = sentences[sentences.find(' ') : right_bound]

                        # get the current index in these sentences
                        curr_index_left = 50 - left_bound
                        curr_index_right = 50 - left_bound + len(confusion_word)

                        # find the best alternative_word
                        min_word, min_perp = confusion_word, sys.maxint
                        for alternative_word in confusion_dic[confusion_word]:
                            alternative_sentences = sentences[0 : curr_index_left] + alternative_word + " " +\
                                                    sentences[curr_index_right + 1:]

                            perp = gt_nragms[topic].generate_perplexity(n, preprocess.preprocess_text(alternative_sentences))

                            if perp < min_perp:
                                min_perp = perp
                                min_word = alternative_word

                        # replace the word with best word
                        text = text[:replace_left] + min_word + text[replace_right:]
                        begin_index = replace_right

                open(corrected_f,'w').write(text)


def main():
    spell_checking_gt_gram()

if __name__ == "__main__":
    main()
