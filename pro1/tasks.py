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

    # get gt_ngram for each topic and read all test data
    ratio = 0.8
    li_ngrams, train_text = {}, {}#key: topic
    split_train_test(ratio = ratio)
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        train_text[topic] = open(train_f, 'r').read()
        li_ngrams[topic] = li_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy, r = {}, [0, 0, 0, 0, 0]
    diff = 0.2
    for i in xrange(0, int(1 / diff)):
        for j in xrange(0, int(1 / diff) - i):
            r[-3] = i * diff
            r[-2] = j * diff
            r[-1] = 1 - r[-2] - r[-3]

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
                            perp = li_ngrams[topic].generate_perplexity(5, text, r = r)

                            if perp < min_perp:
                                min_perp = perp
                                min_topic = topic

                        if label_topic == min_topic:
                            correct += 1
                        _sum += 1

            accuracy[tuple(r)] = 1.0 * correct / _sum
            print "{} {}".format(tuple(r), accuracy[tuple(r)])


    #choose the best r
    # r = [0.001, 0.004, 0.015, 0.03, 0.95] # max(accuracy.iteritems(), key = operator.itemgetter(1))[0]
    # r = list(r_tuple)
    # print "Best: {}: {}".format(list(r_tuple), accuracy[r_tuple])

    # get the result for files in test_for_classification directory
    # test_dir = indir_pre + "data/classification_task/test_for_classification"
    # csv_f = indir_pre + "data/classification_task/li_result.csv"
    #
    # with open(csv_f, 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
    #     writer.writeheader()
    #
    #     for root, dirs, filenames in os.walk(test_dir):
    #         for f in filenames:
    #             text = preprocess.preprocess_file(os.path.join(root, f))
    #             min_perp, min_topic = sys.maxint, ''
    #
    #             for topic in topics:
    #                 perp = li_ngrams[topic].generate_perplexity(n, text, r)
    #                 if perp < min_perp:
    #                     min_perp = perp
    #                     min_topic = topic
    #
    #             writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


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

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in bo_ngram
    for i in xrange(1, 10):
        for j in xrange(1, 10):
            for k in xrange(0, 10):
                r = []
                r.append(round(i * 0.01, 2))
                r.append(round(j * 0.01 + 0.03, 2))
                r.append(round(k * 0.01 + 0.2, 2))
                r.append(1)
                print "r is {}".format(r)
                _sum, correct = 0, 0
                for label_topic, text in test_text.items():
                    sentences = text.split('</s>')
                    for sentence in sentences:
                        sentence += ' </s>'
                        min_perp, min_topic = sys.maxint, label_topic

                        for topic in topics:
                            perp = bo_ngrams[topic].generate_perplexity(4, sentence, r = r)
                            if perp < min_perp:
                                min_perp = perp
                                min_topic = topic

                        if label_topic == min_topic:
                            correct += 1
                        _sum += 1

                accuracy[tuple(r)] = 1.0 * correct / _sum
                print "{}: {}".format(r, accuracy[tuple(r)])

    # for i in xrange(3,4):
    #     _sum, correct = 0, 0
    #     for label_topic, text in test_text.items():
    #         sentences = text.split('</s>')
    #         for sentence in sentences:
    #             sentence += ' </s>'
    #             min_perp, min_topic = sys.maxint, label_topic

    #             for topic in topics:
    #                 perp = bo_ngrams[topic].generate_perplexity(i, sentence)
    #                 if perp < min_perp:
    #                     min_perp = perp
    #                     min_topic = topic

    #             if label_topic == min_topic:
    #                 correct += 1
    #             _sum += 1

    #     accuracy[i] = 1.0 * correct / _sum
    #     print "[{}-gram] {}".format(i, accuracy[i])
    # choose the best n
    n = 3

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/bo_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = bo_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def spell_checker_gt_nrgam(method = 'perplexity'):
    gt_ngrams, train_text, test_text, test_compare = {}, {}, {}, {} #key: topic

    indir_words = indir_pre + "data/spell_checking_task/confusion_set.txt"
    confused_words = open(indir_words, 'r').read().decode("utf-8-sig").encode("utf-8")
    #in the txt, the phrase "may be" confuses split(). for simplicity, we picked it out.
    #create two dictionaries for word checking
    confused_words.replace("may be", "maybe")
    confused_words = confused_words.split()
    words = {}
    changed_text = ""
    words["even"] = dict()
    words["odd"] = dict()
    end = False;
    for i in xrange(0, len(confused_words)-1, 2):
        words["even"][confused_words[i]]=[]
        words["odd"][confused_words[i+1]]=[]

    for i in xrange(0, len(confused_words)-1, 2):
        words["even"][confused_words[i]].append(confused_words[i+1])
        words["odd"][confused_words[i+1]].append(confused_words[i])

    words["even"]["maybe"] = ["may be"]

    for side in words:

        for indexw in words[side]:
            words[side][indexw].append("")

    for topic in topics:
        gt_ngrams[topic] = dict()
        test_text[topic] = dict()
        test_compare[topic] = dict()
        for i in xrange(1, 5):
            gt_ngrams[topic][i] = dict()


    #create training and testing text
    for topic in topics:
        train_f = indir_pre + "data/spell_checking_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/spell_checking_task/{}/train_modified_docs".format(topic)
        test_cf = indir_pre + "data/spell_checking_task/{}/train_docs".format(topic)

        num_file = len(os.listdir(test_f))
        num_train_file = math.floor(num_file * 0.8)

        for root, dirs, filenames in os.walk(test_f):
            for i, f in enumerate(filenames):
                if i > num_train_file:
                    raw_content = preprocess.preprocess_file(os.path.join(root, f),"sentences")
                    #test_text[topic][f] = dict()
                    test_text[topic][f] = raw_content
                    raw_content = preprocess.preprocess_file(os.path.join(test_cf, f.replace("_modified", "")),"sentences")
                    test_compare[topic][f.replace("_modified", "")] = raw_content
        if not os.path.isfile(train_f):
            split_train_test('spell_checking_task')
        train_text[topic] = open(train_f, 'r').read()
        for i in xrange(1, 5):
            gt_ngrams[topic][i] = gt_ngram(train_text[topic])

    #generate good-turing n-gram for the training set
    for i in xrange(1, 5):
        for label_topic, text in train_text.items():
            gt_ngrams[topic][i].npro_dict = gt_ngrams[topic][i].generate_ngram(i)

    #spell check and accuracy calculation
    #initiate variables
    correct_rate = {}
    correct = {}
    sentence_count = {}
    compare_count = {}
    compare_rate = {}
    for i in xrange(1, 5):
        sentence_count[i] = 0
        correct[i] = 0
        compare_count[i] = 0
        for topic in topics:
            #each file to be examined under the topic
            for filename in test_text[topic]:
                filename_compare = filename
                filename_compare = filename_compare.replace("_modified","")
                #each sentence in the file
                sen_processing = test_text[topic][filename][0].split()
                for j in xrange (0, len(test_text[topic][filename])):
                    #generate the perplexity of the original sentence
                    perp_origin = gt_ngrams[topic][i].generate_perplexity(i,test_text[topic][filename][j])
                    #two-way dictionary
                    for side in words:
                        for word1 in words[side]:
                            sen_tokens = test_text[topic][filename][j].split()
                            #if there's a confused word
                            if word1 in sen_tokens:
                                #if processing a new sentence, count +1
                                #if sen_tokens != sen_processing:
                                sentence_count[i] += 1
                                #sen_processing = sen_tokens
                                #list of alternative sentences
                                alternative = []
                                #list of perplexity for the alternative sentences
                                perp_alt = []
                                #alternative.append(test_text[topic][filename][j])
                                #for each possible alternate word
                                changed_text = test_text[topic][filename][j]

                                """
                                break the sentence into a list of tokens, replace the token to be examined,
                                reconstruct the list into a string
                                """
                                new_word = word1
                                for k in xrange(0,len(words[side][word1])):
                                    #replace the word of interest in the list of tokens
                                    alternative.append(sen_tokens)
                                    alternative[k] = [w.replace(word1,''.join(words[side][word1][k])) for w in alternative[k]]

                                    #reconstructing the new sentence
                                    sent_Tobe = ""
                                    for m in xrange(0, len(alternative[k])):
                                        if alternative[k][m].isalpha():
                                            sent_Tobe += ' '.join(alternative[k][m])
                                        else:
                                            sent_Tobe += ''.join(alternative[k][m])

                                    #generate the perplexity of the new sentence
                                    if len(sent_Tobe)>0:
                                        perp_alt.append(gt_ngrams[topic][i].generate_perplexity(i,sent_Tobe))
                                    else:
                                        perp_alt.append(1.0*sys.maxint)

                                    #if the new sentence has a lower perplexity
                                    if perp_alt[k] < perp_origin and perp_alt[k] == min(perp_alt):
                                        #replace the old sentence with the new one
                                        #test_text[topic][filename][j]=''.join(alternative[k])
                                        changed_text = sent_Tobe
                                        new_word = words[side][word1][k]

                                if len(new_word) > 0 and new_word in test_compare[topic][filename_compare][j]:
                                #test_compare[topic][filename_compare][j]:
                                    correct[i] += 1
                                else:
                                    if word1 not in test_compare[topic][filename_compare][j]:
                                        correct[i] += 1
                                if test_compare[topic][filename_compare][j] == test_text[topic][filename][j]:
                                    compare_count[i] += 1
                                        #print i, correct[i], sentence_count[i]
                                        #print the new sentence with the new word
        #print correct
        #print sentence_count
        correct_rate[i] = 1.0 * correct[i] / sentence_count[i]
        compare_rate[i] = 1.0 * compare_count[i] / sentence_count[i]
        print "{} gram correct spell check rate = {}".format(i, correct_rate[i])
        print "{} gram correct compare rate = {}".format(i, compare_rate[i])


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
    for i in xrange(1, 5):

        # rank the counter for each topic
        counter_rank_dic = {}
        for topic in topics:
            ngrams[topic].ncounter_dic[i] = ngrams[topic].ncounter_dic[i] if i in ngrams[topic].ncounter_dic else \
                                            ngrams[topic].ntoken_count(i)
            sorted_counter = sorted(ngrams[topic].ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)
            rank = list((k[0]) for k in sorted_counter)
            counter_rank_dic[topic] = dict((rank[k], k) for k in xrange(len(rank)))

        _sum, correct = 1, 0
        max_dis = 1500
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
                    test_counter_rank_list = list((k[0]) for k in sorted_counter)

                    min_dis, min_topic = sys.maxint, label_topic
                    for topic in topics:
                        dis = 0
                        for idx in xrange(len(test_counter_rank_list)):
                            tokens = test_counter_rank_list[idx]
                            if tokens in counter_rank_dic[topic]:
                                dis += min(abs(idx - counter_rank_dic[topic][tokens]), max_dis)
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

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/dis_result.csv"

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
                test_counter_rank_list = list((i[0]) for i in sorted_counter)

                min_dis, min_topic = sys.maxint, label_topic
                for topic in topics:
                    dis = 0
                    for idx in xrange(len(test_counter_rank_list)):
                        tokens = test_counter_rank_list[idx]
                        if tokens in counter_rank_dic[topic]:
                            dis += min(abs(idx - counter_rank_dic[topic][tokens]), max_dis)
                        else:
                            dis += max_dis

                    if dis < min_dis:
                        min_dis = dis
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def topic_classification_ngram_dis_mul():
    # get gt_ngram for each topic and read all test data
    ratio = 0.9
    ngrams, train_text  = {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        split_train_test(ratio = ratio)
        train_text[topic] = open(train_f, 'r').read()
        ngrams[topic] = ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    counter_rank_dic = {}
    for i in xrange(1, 3):

        # rank the counter for each topic
        counter_rank_dic[i] = {}
        for topic in topics:
            ngrams[topic].ncounter_dic[i] = ngrams[topic].ncounter_dic[i] if i in ngrams[topic].ncounter_dic else \
                                            ngrams[topic].ntoken_count(i)
            sorted_counter = sorted(ngrams[topic].ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)
            rank = list((k[0]) for k in sorted_counter)
            counter_rank_dic[i][topic] = dict((rank[k], k) for k in xrange(len(rank)))


    r = [0, 0]
    for j in xrange(10):
        r[0] = j * 0.1
        r[1] = 1 - r[0]
        _sum, correct = 0, 0
        max_dis = [1500, 1800]
        for label_topic in topics:
            test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
            for root, dirs, filenames in os.walk(test_dir):
                for idx, f in enumerate(filenames):
                    if idx < len(filenames) * ratio:
                        continue
                    test_counter_rank_list = {}
                    for i in xrange(1, 3):
                        text = open(os.path.join(root, f),'r').read()
                        test_ngram = ngram(preprocess.preprocess_text(text))
                        test_ngram.ncounter_dic[i] = test_ngram.ncounter_dic[i] if i in test_ngram.ncounter_dic else \
                                                     test_ngram.ntoken_count(i)
                        sorted_counter = sorted(test_ngram.ncounter_dic[i].items(), key = lambda x: x[1], reverse = True)
                        test_counter_rank_list[i] = list((k[0]) for k in sorted_counter)


                    min_dis, min_topic = sys.maxint, label_topic
                    for topic in topics:
                        dis = 0
                        for i in xrange(1, 3):
                            for idx in xrange(len(test_counter_rank_list[i])):
                                tokens = test_counter_rank_list[i][idx]
                                if tokens in counter_rank_dic[i][topic]:
                                    dis += min(abs(idx - counter_rank_dic[i][topic][tokens]), max_dis[i - 1])
                                else:
                                    dis += max_dis[i - 1]

                            if dis < min_dis:
                                min_dis = dis
                                min_topic = topic

                    if label_topic == min_topic:
                        correct += 1
                    _sum += 1

        accuracy[tuple(r)] = 1.0 * correct / _sum
        print "{}: {}".format(r, accuracy[tuple(r)])

# TODO: change name and comment
def topic_classification_ngram_dis_1():
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

def spell_checking_jiao():
    

def main():
    topic_classification_ngram_dis_1()


if __name__ == "__main__":
    main()
