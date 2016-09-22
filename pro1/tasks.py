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
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        gt_ngrams[topic] = gt_ngram(content)

        print "\nTopic: {}".format(topic)
        for i in xrange(1, 6):
            print "[{}-gram]: {}".format(i, gt_ngrams[topic].generate_perplexity(i, content))


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
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        gt_ngrams[topic] = gt_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    for i in xrange(1, 5):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(i, sentence)
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
        pointer = int(len(tokens) * 1.0)
        # while tokens[pointer] != '</s>':
        #     pointer += 1

        train_text = ' '.join(tokens)
        # test_text = ' '.join(tokens[(pointer+2):])

        train_path = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        # test_path = indir_pre + "data/classification_task/{}/test.txt".format(topic)
        open(train_path, 'w').write(train_text)
        # open(test_path, 'w').write(test_text)


def topic_classification_li_ngram():
    # TODO when li_gram done, test
    # get gt_ngram for each topic and read all test data
    li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        # test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        # if not os.path.isfile(train_f): # or not os.path.isfile(test_f):
        split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        # test_text[topic] = open(test_f, 'r').read()

        li_ngrams[topic] = li_ngram(train_text[topic])
    r2test = [[0.001, 0.004, 0.015, 0.03, 0.95]]
    accuracy, r = {}, [0,0,0,0]
    # for i in xrange(1, 10):
    #     for j in xrange(1, 10):
    #         for k in xrange(0, 10):

    #             r[0] = round(i * 0.01, 2)
    #             r[1] = round(j * 0.01 + 0.03, 2)
    #             r[2] = round(k * 0.01 + 0.05, 2)
    #             r[3] = round(1 - r[0] - r[1] - r[2], 2)
    # setting ngram number
    n = 5

    # start with testing
    # for r in r2test:
    #     print "r is {}".format(r) 
    #     _sum, correct = 0, 0
    #     for label_topic, text in test_text.items():
    #         sentences = text.split('</s>')
    #         for sentence in sentences:
    #             sentence += ' </s>'
    #             min_perp, min_topic = sys.maxint, label_topic

    #             for topic in topics:
    #                 perp = li_ngrams[topic].generate_perplexity(n, sentence, r = r)
    #                 if perp < min_perp:
    #                     min_perp = perp
    #                     min_topic = topic

    #             if label_topic == min_topic:
    #                 correct += 1
    #             _sum += 1

    #     accuracy[tuple(r)] = 1.0 * correct / _sum
    #     print "{}: {}".format(r, accuracy[tuple(r)])

    #choose the best r
    r = [0.001, 0.004, 0.015, 0.03, 0.95] # max(accuracy.iteritems(), key = operator.itemgetter(1))[0]
    # r = list(r_tuple)
    # print "Best: {}: {}".format(list(r_tuple), accuracy[r_tuple])

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
                    perp = li_ngrams[topic].generate_perplexity(n, text, r)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def generate_perplexity_bo_ngram():
    bo_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        bo_ngrams[topic] = bo_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, bo_ngrams[topic].generate_perplexity(3, content))


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

def generate_perplexity_bo_ngram():
    bo_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        bo_ngrams[topic] = bo_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, bo_ngrams[topic].generate_perplexity(3, content))

def generate_perplexity_li_ngram():
    li_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        li_ngrams[topic] = li_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, li_ngrams[topic].generate_perplexity(3, content))

# def topic_classification_li_ngram():
#     """
#     calculate the accuracy for topic classification with different
#     n in Good-Turing ngram, then choose the best one to classify files
#     in test_for_classification directory, and write results into
#     li_result.csv in classification_task directory
#     """

#     # get li_ngram for each topic and read all test data
#     li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
#     for topic in topics:
#         train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
#         test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
#         if not os.path.isfile(train_f) or not os.path.isfile(test_f):
#             split_train_test()

#         train_text[topic] = open(train_f, 'r').read()
#         test_text[topic] = open(test_f, 'r').read()

#         li_ngrams[topic] = li_ngram(train_text[topic])

#     # calculate the accuracy for n-gram and choose the best one
#     accuracy = {} # key: the n in li_ngram
#     for i in xrange(3,4):
#         _sum, correct = 0, 0
#         for label_topic, text in test_text.items():
#             sentences = text.split('</s>')
#             for sentence in sentences:
#                 sentence += ' </s>'
#                 min_perp, min_topic = sys.maxint, label_topic

#                 for topic in topics:
#                     perp = li_ngrams[topic].generate_perplexity(i, sentence)
#                     if perp < min_perp:
#                         min_perp = perp
#                         min_topic = topic

#                 if label_topic == min_topic:
#                     correct += 1
#                 _sum += 1

#         accuracy[i] = 1.0 * correct / _sum
#         print "[{}-gram] {}".format(i, accuracy[i])
#     #choose the best n
#     n = 3

#     # get the result for files in test_for_classification directory
#     test_dir = indir_pre + "data/classification_task/test_for_classification"
#     csv_f = indir_pre + "data/classification_task/li_result.csv"

#     with open(csv_f, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
#         writer.writeheader()

#         for root, dirs, filenames in os.walk(test_dir):
#             for f in filenames:
#                 text = preprocess.preprocess_file(os.path.join(root, f))
#                 min_perp, min_topic = sys.maxint, ''

#                 for topic in topics:
#                     perp = li_ngrams[topic].generate_perplexity(n, text)
#                     if perp < min_perp:
#                         min_perp = perp
#                         min_topic = topic

#                 writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def spell_checker_gt_nrgam():
    pass


def main():
    topic_classification_li_ngram()


if __name__ == "__main__":
    main()
