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
    gt_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f):
            split_train_test()
        train_text[topic] = open(train_f, 'r').read()
        gt_ngrams[topic] = gt_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    for i in xrange(1, 8):
        _sum, correct = 0, 0
        for label_topic in topics:
            test_dir = indir_pre + "data/classification_task/{}".format(label_topic)
            for root, dirs, filenames in os.walk(test_dir):
                for idx, f in enumerate(filenames):
                    if idx < len(filenames) * 0.8:
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


def split_train_test():
    """
    split train_docs into     training:test = 4:1
    store the preprocessed file train.txt and test.txt in each topic directory
    """
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        tokens = content.split()

        # find the nearest </s> after 80% content
        pointer = int(len(tokens) * 0.8)
        while tokens[pointer] != '</s>':
            pointer += 1

        train_text = ' '.join(tokens[:(pointer+1)])
        test_text = ' '.join(tokens[(pointer+2):])

        train_path = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_path = indir_pre + "data/classification_task/{}/test.txt".format(topic)
        open(train_path, 'w').write(train_text)
        open(test_path, 'w').write(test_text)


def topic_classification_li_ngram():
    # TODO when li_gram done, test
    # get gt_ngram for each topic and read all test data
    li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        li_ngrams[topic] = li_ngram(train_text[topic])

    accuracy, r = {}, []
    for i in xrange(0, 11):
        for j in xrange(0, 11 - i):
            r[0] = round(i * 0.1, 1)
            r[1] = round(j * 0.1, 1)
            r[2] = round(1 - r[0] - r[1], 1)

            _sum, correct = 0, 0
            for label_topic, text in test_text.items():
                sentences = text.split('</s>')
                for sentence in sentences:
                    sentence += ' </s>'
                    min_perp, min_topic = sys.maxint, label_topic

                    for topic in topics:
                        perp = li_ngrams[topic].generate_perplexity(3, sentence, r)
                        if perp < min_perp:
                            min_perp = perp
                            min_topic = topic

                    if label_topic == min_topic:
                        correct += 1
                    _sum += 1

            accuracy[tuple(r)] = 1.0 * correct / _sum
            print "{}: {}".format(r, accuracy[tuple(r)])

    #choose the best r
    r_tuple = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]
    r = list(r_tuple)
    print "Best: {}: {}".format(list(r_tuple), accuracy[r_tuple])

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/li_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(n, text, r)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


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
    for i in xrange(3,4):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = bo_ngrams[topic].generate_perplexity(i, sentence)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])
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


def topic_classification_li_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    li_result.csv in classification_task directory
    """

    # get li_ngram for each topic and read all test data
    li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        li_ngrams[topic] = li_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in li_ngram
    for i in xrange(3,4):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = li_ngrams[topic].generate_perplexity(i, sentence)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])
    #choose the best n
    n = 3

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/li_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = li_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def spell_checker_gt_nrgam():
    pass



def main():
    topic_classification_gt_ngram()


if __name__ == "__main__":
    main()
